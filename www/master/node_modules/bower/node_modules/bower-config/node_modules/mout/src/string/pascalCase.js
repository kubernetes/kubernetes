define(['../lang/toString', './camelCase', './upperCase'], function(toString, camelCase, upperCase){
    /**
     * camelCase + UPPERCASE first char
     */
    function pascalCase(str){
        str = toString(str);
        return camelCase(str).replace(/^[a-z]/, upperCase);
    }

    return pascalCase;
});
