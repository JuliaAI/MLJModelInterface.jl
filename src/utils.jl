if VERSION < v"1.1"
    fieldtypes(t) = Tuple(fieldtype(t, i) for i = 1:fieldcount(t))
end

function coretype(M)
    if isdefined(M, :name)
        return M.name
    else
        return coretype(M.body)
    end
end
