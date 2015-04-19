define(['./createObject'], function(createObject){

    /**
    * Inherit prototype from another Object.
    * - inspired by Nicholas Zackas <http://nczonline.net> Solution
    * @param {object} child Child object
    * @param {object} parent    Parent Object
    */
    function inheritPrototype(child, parent){
        var p = createObject(parent.prototype);
        p.constructor = child;
        child.prototype = p;
        child.super_ = parent;
        return p;
    }

    return inheritPrototype;
});
