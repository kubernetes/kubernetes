define(['../lang/toString', './lowerCase', './upperCase'], function(toString, lowerCase, upperCase){
    /**
     * UPPERCASE first char of each word.
     */
    function properCase(str){
        str = toString(str);
        return lowerCase(str).replace(/^\w|\s\w/g, upperCase);
    }

    return properCase;
});
