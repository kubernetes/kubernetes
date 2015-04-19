define(['../math/countSteps'], function (countSteps) {

    /**
     * Returns an Array of numbers inside range.
     */
    function range(start, stop, step) {
        if (stop == null) {
            stop = start;
            start = 0;
        }
        step = step || 1;

        var result = [],
            nSteps = countSteps(stop - start, step),
            i = start;

        while (i <= stop) {
            result.push(i);
            i += step;
        }

        return result;
    }

    return range;

});
