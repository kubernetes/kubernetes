define(['../array/max', './values'], function(arrMax, values) {

    /**
     * Returns maximum value inside object.
     */
    function max(obj, compareFn) {
        return arrMax(values(obj), compareFn);
    }

    return max;
});
