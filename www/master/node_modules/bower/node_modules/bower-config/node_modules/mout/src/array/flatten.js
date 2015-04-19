define(['../lang/isArray', './append'], function (isArray, append) {

    /*
     * Helper function to flatten to a destination array.
     * Used to remove the need to create intermediate arrays while flattening.
     */
    function flattenTo(arr, result, level) {
        if (arr == null) {
            return result;
        } else if (level === 0) {
            append(result, arr);
            return result;
        }

        var value,
            i = -1,
            len = arr.length;
        while (++i < len) {
            value = arr[i];
            if (isArray(value)) {
                flattenTo(value, result, level - 1);
            } else {
                result.push(value);
            }
        }
        return result;
    }

    /**
     * Recursively flattens an array.
     * A new array containing all the elements is returned.
     * If `shallow` is true, it will only flatten one level.
     */
    function flatten(arr, level) {
        level = level == null? -1 : level;
        return flattenTo(arr, [], level);
    }

    return flatten;

});

