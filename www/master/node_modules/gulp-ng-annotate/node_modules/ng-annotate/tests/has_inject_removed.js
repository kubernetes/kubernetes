
// @ngInject
// already has .$inject array (before function definition)
function Ctrl1(a) {
}

// @ngInject
// already has .$inject array (after function definition)
function Ctrl2(a) {
}

function outer() {
    return {
        controller: MyCtrl,
    };

    // @ngInject
    function MyCtrl(a) {
    }
}
