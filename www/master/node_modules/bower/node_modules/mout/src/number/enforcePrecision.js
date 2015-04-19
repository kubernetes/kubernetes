define(['../lang/toNumber'], function(toNumber){
    /**
     * Enforce a specific amount of decimal digits and also fix floating
     * point rounding issues.
     */
    function enforcePrecision(val, nDecimalDigits){
        val = toNumber(val);
        var pow = Math.pow(10, nDecimalDigits);
        return +(Math.round(val * pow) / pow).toFixed(nDecimalDigits);
    }
    return enforcePrecision;
});
