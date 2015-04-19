define(['./toInt', './nth'], function (toInt, nth) {

    /**
     * converts number into ordinal form (1st, 2nd, 3rd, 4th, ...)
     */
    function ordinal(n){
       n = toInt(n);
       return n + nth(n);
    }

    return ordinal;

});
