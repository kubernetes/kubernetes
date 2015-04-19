define(['./some'], function (some) {

    /**
     * Check if object contains value
     */
    function contains(obj, needle) {
        return some(obj, function(val) {
            return (val === needle);
        });
    }
    return contains;

});
