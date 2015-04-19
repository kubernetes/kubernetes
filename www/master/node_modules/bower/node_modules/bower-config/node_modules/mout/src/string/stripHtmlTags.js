define(['../lang/toString'], function(toString){
    /**
     * Remove HTML tags from string.
     */
    function stripHtmlTags(str){
        str = toString(str);

        return str.replace(/<[^>]*>/g, '');
    }
    return stripHtmlTags;
});
