define(['../lang/toString', './trim'], function(toString, trim){
    /**
     * Limit number of chars.
     */
    function truncate(str, maxChars, append, onlyFullWords){
        str = toString(str);
        append = append || '...';
        maxChars = onlyFullWords? maxChars + 1 : maxChars;

        str = trim(str);
        if(str.length <= maxChars){
            return str;
        }
        str = str.substr(0, maxChars - append.length);
        //crop at last space or remove trailing whitespace
        str = onlyFullWords? str.substr(0, str.lastIndexOf(' ')) : trim(str);
        return str + append;
    }
    return truncate;
});
