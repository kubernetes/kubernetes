define(['./map'], function (map) {

    /**
     * Extract a list of property values.
     */
    function pluck(arr, propName){
        return map(arr, propName);
    }

    return pluck;

});
