define(function () {
    /**
     * Round number to a specific radix
     */
    function round(value, radix){
        radix = radix || 1; // default round 1
        return Math.round(value / radix) * radix;
    }

    return round;

});
