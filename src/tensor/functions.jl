# SPDX-License-Identifier: MIT

export @generate_fastmatmul

"""
    @generate_fastmatmul alg name type
"""
macro generate_fastmatmul( alg, name, type )
    return quote
        eval( _generate_fastmatmul( $alg, $name, $type ) )
    end
end

"""
    _generate_fastmatmul( tfmm::TensorFMMAlgorithm, name::String; etype = Float64 )
"""
function _generate_fastmatmul( tfmm::TensorFMMAlgorithm, name::String; etype = Float64 )
    # Function signature
    funcsig  = :( $(Symbol(name))( A::Matrix{T}, B::Matrix{T} ) where {T <: $etype} )

    # Create the main math portion and the return statement
    mathexpr = _generate_fastmatmul_expr( tfmm, etype = etype )
    returns  = Expr( :block, :(return C) )

    # Assemble the function for the multiplication
    funcbody = Expr( :block, mathexpr, returns )
    Expr( :function, funcsig, funcbody )
end

function _generate_fastmatmul_expr( tfmm::TensorFMMAlgorithm; etype = Float64 )
    # Transform the matrices to row-linear ordering
    init = quote
                C = Matrix{$(etype)}(undef, $(tfmm.n), $(tfmm.p) )
                Ap = PermutedDimsArray( A, (2, 1) )
                Bp = PermutedDimsArray( B, (2, 1) )
                Cp = PermutedDimsArray( C, (2, 1) )
           end

    # Form the partial products
    partial = Vector{Expr}()
    for (i, u) in enumerate( eachcol( tfmm.U ) )
        Aadder = _adder_chain( u, "Ap", true )
        Badder = _adder_chain( tfmm.V[:,i], "Bp", true )

        push!( partial, _partial_product( Aadder, Badder, i ) )
    end

    # Form the final additions
    final = Vector{Expr}()
    for (i, w) in enumerate( eachrow( tfmm.W ) )
        Cadder = _adder_chain( w, "m", false )

        push!( final, :(Cp[$i] = $Cadder) )
    end

    partials = Expr( :block, partial...)
    finals   = Expr( :block, final...)

    # Form the full math expression
    Expr( :block, init, partials, finals )
end

function _partial_product( u, v, varnum::Int )
    mname = Symbol( "m$varnum" )

    if ( u == :() ) || ( v == :() )
        return :($mname = 0)
    end

    return :($mname = $u * $v)
end

function _adder_chain( u, varname::String, isarray::Bool )
    varScaled = Vector{Tuple{Expr,Bool}}()

    adder = nothing

    for (ind, val) in enumerate( u )
        # Don't include multiplications by zero
        iszero( val ) && continue

        if isarray
            indexpr = :($(Symbol(varname))[$ind])
        else
            indexpr = Symbol("$varname$ind")
        end

        # Don't do the multiplications by one
        if isone( abs( val ) )
            varScaled = indexpr
        else
            varScaled = :($(abs(val)) * $indexpr)
        end

        isneg = signbit( val )

        # Split everything out to get better looking functions when printed in the terminal
        if isnothing( adder )
            if isneg
                adder = :(-$varScaled)
            else
                adder = :($varScaled)
            end
        else
            if isneg
                adder = :($adder - $varScaled)
            else
                adder = :($adder + $varScaled)
            end
        end
    end

    return isnothing( adder ) ? :() : adder
end
