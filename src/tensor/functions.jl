# SPDX-License-Identifier: MIT

"""
    @generate_fastmatmul alg name type
"""
macro generate_fastmatmul( alg, name, type )
    return quote
        eval( _generate_fastmatmul( $alg, $name, $type ) )
    end
end


"""
    _generate_fastmatmul( tfmm::TensorFMMAlgorithm, name::String, T )
"""
function _generate_fastmatmul( tfmm::TensorFMMAlgorithm, name::String, T )
    # Initial part of the function is transforming to row-linear ordering
    init = quote
                C = Matrix{$(T)}(undef, $(tfmm.n), $(tfmm.p) )
                Ap = PermutedDimsArray( A, (2, 1) )
                Bp = PermutedDimsArray( B, (2, 1) )
                Cp = PermutedDimsArray( C, (2, 1) )
           end

    # Form the partial products
    partial = Vector{Expr}()
    for (i, u) in enumerate( eachcol( tfmm.U ) )
        Aadder = _adder_chain( u, "Ap", true )
        Badder = _adder_chain( tfmm.V[:,i], "Bp", true )

        mname = Symbol( "m$i" )
        push!( partial, :($mname = $Aadder * $Badder) )
    end

    # Form the final additions
    final = Vector{Expr}()
    for (i, w) in enumerate( eachrow( tfmm.W ) )
        Cadder = _adder_chain( w, "m", false )

        push!( final, :(Cp[$i] = $Cadder) )
    end

    partials = Expr( :block, partial...)
    finals   = Expr( :block, final...)
    returns  = Expr( :block, :(return C) )

    # Assemble the function for the multiplication
    funcsig  = :( $(Symbol(name))( A::Matrix{T}, B::Matrix{T} ) where {T <: $T} )
    funcbody = Expr( :block, init, partials, finals, returns )

    Expr( :function, funcsig, funcbody )
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
            varScaled = :($val * $indexpr)
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

    return adder
end
