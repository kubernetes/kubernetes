define(['./forIn'], function (forIn) {

    /**
     * return a list of all enumerable properties that have function values
     */
    function functions(obj){
        var keys = [];
        forIn(obj, function(val, key){
            if (typeof val === 'function'){
                keys.push(key);
            }
        });
        return keys.sort();
    }

    return functions;

});
