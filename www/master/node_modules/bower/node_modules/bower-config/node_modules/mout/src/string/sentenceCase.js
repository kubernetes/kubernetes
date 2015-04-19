define(['../lang/toString', './lowerCase', './upperCase'], function(toString, lowerCase, upperCase){
    /**
     * UPPERCASE first char of each sentence and lowercase other chars.
     */
    function sentenceCase(str){
        str = toString(str);

        // Replace first char of each sentence (new line or after '.\s+') to
        // UPPERCASE
        return lowerCase(str).replace(/(^\w)|\.\s+(\w)/gm, upperCase);
    }
    return sentenceCase;
});
