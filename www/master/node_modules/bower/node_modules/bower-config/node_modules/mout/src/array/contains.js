define(['./indexOf'], function (indexOf) {

    /**
     * If array contains values.
     */
    function contains(arr, val) {
        return indexOf(arr, val) !== -1;
    }
    return contains;
});
