define(['../lang/toString'], function(toString){
    /**
     * "Safer" String.toUpperCase()
     */
    function upperCase(str){
        str = toString(str);
        return str.toUpperCase();
    }
    return upperCase;
});
