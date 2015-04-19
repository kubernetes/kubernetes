(function (tree) {

tree.Selector = function (elements, extendList, condition, index, currentFileInfo, isReferenced) {
    this.elements = elements;
    this.extendList = extendList;
    this.condition = condition;
    this.currentFileInfo = currentFileInfo || {};
    this.isReferenced = isReferenced;
    if (!condition) {
        this.evaldCondition = true;
    }
};
tree.Selector.prototype = {
    type: "Selector",
    accept: function (visitor) {
        if (this.elements) {
            this.elements = visitor.visitArray(this.elements);
        }
        if (this.extendList) {
            this.extendList = visitor.visitArray(this.extendList);
        }
        if (this.condition) {
            this.condition = visitor.visit(this.condition);
        }
    },
    createDerived: function(elements, extendList, evaldCondition) {
        evaldCondition = (evaldCondition != null) ? evaldCondition : this.evaldCondition;
        var newSelector = new(tree.Selector)(elements, extendList || this.extendList, null, this.index, this.currentFileInfo, this.isReferenced);
        newSelector.evaldCondition = evaldCondition;
        newSelector.mediaEmpty = this.mediaEmpty;
        return newSelector;
    },
    match: function (other) {
        var elements = this.elements,
            len = elements.length,
            olen, i;

        other.CacheElements();

        olen = other._elements.length;
        if (olen === 0 || len < olen) {
            return 0;
        } else {
            for (i = 0; i < olen; i++) {
                if (elements[i].value !== other._elements[i]) {
                    return 0;
                }
            }
        }

        return olen; // return number of matched elements
    },
    CacheElements: function(){
        var css = '', len, v, i;

        if( !this._elements ){

            len = this.elements.length;
            for(i = 0; i < len; i++){

                v = this.elements[i];
                css += v.combinator.value;

                if( !v.value.value ){
                    css += v.value;
                    continue;
                }

                if( typeof v.value.value !== "string" ){
                    css = '';
                    break;
                }
                css += v.value.value;
            }

            this._elements = css.match(/[,&#\*\.\w-]([\w-]|(\\.))*/g);

            if (this._elements) {
                if (this._elements[0] === "&") {
                    this._elements.shift();
                }

            } else {
                this._elements = [];
            }

        }
    },
    isJustParentSelector: function() {
        return !this.mediaEmpty && 
            this.elements.length === 1 && 
            this.elements[0].value === '&' && 
            (this.elements[0].combinator.value === ' ' || this.elements[0].combinator.value === '');
    },
    eval: function (env) {
        var evaldCondition = this.condition && this.condition.eval(env),
            elements = this.elements, extendList = this.extendList;

        elements = elements && elements.map(function (e) { return e.eval(env); });
        extendList = extendList && extendList.map(function(extend) { return extend.eval(env); });

        return this.createDerived(elements, extendList, evaldCondition);
    },
    genCSS: function (env, output) {
        var i, element;
        if ((!env || !env.firstSelector) && this.elements[0].combinator.value === "") {
            output.add(' ', this.currentFileInfo, this.index);
        }
        if (!this._css) {
            //TODO caching? speed comparison?
            for(i = 0; i < this.elements.length; i++) {
                element = this.elements[i];
                element.genCSS(env, output);
            }
        }
    },
    toCSS: tree.toCSS,
    markReferenced: function () {
        this.isReferenced = true;
    },
    getIsReferenced: function() {
        return !this.currentFileInfo.reference || this.isReferenced;
    },
    getIsOutput: function() {
        return this.evaldCondition;
    }
};

})(require('../tree'));
