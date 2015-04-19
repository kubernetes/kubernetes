'use strict';

function arrayRemove(array, value) {
    var index = array.indexOf(value);

    if (index !== -1) {
        array.splice(index, 1);
    }

    return array;
}

module.exports = arrayRemove;
