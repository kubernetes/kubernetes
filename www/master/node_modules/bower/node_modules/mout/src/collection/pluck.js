define(['./map'], function (map) {

    /**
     * Extract a list of property values.
     */
    function pluck(list, key) {
        return map(list, function(value) {
            return value[key];
        });
    }

    return pluck;

});
