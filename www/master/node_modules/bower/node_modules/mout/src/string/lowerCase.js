define(['../lang/toString'], function(toString){
    /**
     * "Safer" String.toLowerCase()
     */
    function lowerCase(str){
        str = toString(str);
        return str.toLowerCase();
    }

    return lowerCase;
});
