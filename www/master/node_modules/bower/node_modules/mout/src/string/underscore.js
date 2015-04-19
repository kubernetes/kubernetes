define(['../lang/toString', './slugify', './unCamelCase'], function(toString, slugify, unCamelCase){
    /**
     * Replaces spaces with underscores, split camelCase text, remove non-word chars, remove accents and convert to lower case.
     */
    function underscore(str){
        str = toString(str);
        str = unCamelCase(str);
        return slugify(str, "_");
    }
    return underscore;
});
