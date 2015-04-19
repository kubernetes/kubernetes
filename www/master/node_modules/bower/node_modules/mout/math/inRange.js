
    /**
    * Checks if value is inside the range.
    */
    function inRange(val, min, max, threshold){
        threshold = threshold || 0;
        return (val + threshold >= min && val - threshold <= max);
    }

    module.exports = inRange;

