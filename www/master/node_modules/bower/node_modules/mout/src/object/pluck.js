define(['./map', '../function/prop'], function (map, prop) {

    /**
     * Extract a list of property values.
     */
    function pluck(obj, propName){
        return map(obj, prop(propName));
    }

    return pluck;

});
