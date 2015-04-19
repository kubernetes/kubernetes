define(['../lang/toString', './WHITE_SPACES', './ltrim', './rtrim'], function(toString, WHITE_SPACES, ltrim, rtrim){
    /**
     * Remove white-spaces from beginning and end of string.
     */
    function trim(str, chars) {
        str = toString(str);
        chars = chars || WHITE_SPACES;
        return ltrim(rtrim(str, chars), chars);
    }

    return trim;
});
