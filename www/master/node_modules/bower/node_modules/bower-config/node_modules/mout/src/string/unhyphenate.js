define(['../lang/toString'], function(toString){
    /**
     * Replaces hyphens with spaces. (only hyphens between word chars)
     */
    function unhyphenate(str){
        str = toString(str);
        return str.replace(/(\w)(-)(\w)/g, '$1 $3');
    }
    return unhyphenate;
});
