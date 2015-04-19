define(['./isNumber', '../number/isNaN'], function (isNumber, $isNaN) {

    /**
     * Check if value is NaN for realz
     */
    function isNaN(val){
        // based on the fact that NaN !== NaN
        // need to check if it's a number to avoid conflicts with host objects
        // also need to coerce ToNumber to avoid edge case `new Number(NaN)`
        return !isNumber(val) || $isNaN(Number(val));
    }

    return isNaN;

});
