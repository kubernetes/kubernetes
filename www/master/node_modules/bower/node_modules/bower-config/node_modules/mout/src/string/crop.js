define(['../lang/toString', './truncate'], function (toString, truncate) {
    /**
     * Truncate string at full words.
     */
     function crop(str, maxChars, append) {
         str = toString(str);
         return truncate(str, maxChars, append, true);
     }

     return crop;
});
