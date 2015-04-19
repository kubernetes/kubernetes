define(['../array/min', './values'], function(arrMin, values) {

    /**
     * Returns minimum value inside object.
     */
    function min(obj, iterator) {
        return arrMin(values(obj), iterator);
    }

    return min;
});
