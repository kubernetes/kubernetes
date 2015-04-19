define(['../lang/toString', './slugify', './unCamelCase'], function(toString, slugify, unCamelCase){
    /**
     * Replaces spaces with hyphens, split camelCase text, remove non-word chars, remove accents and convert to lower case.
     */
    function hyphenate(str){
        str = toString(str);
        str = unCamelCase(str);
        return slugify(str, "-");
    }

    return hyphenate;
});
