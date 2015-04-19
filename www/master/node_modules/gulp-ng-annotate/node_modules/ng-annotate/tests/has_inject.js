Ctrl1.$inject = ["serviceName"];
// @ngInject
// already has .$inject array (before function definition)
function Ctrl1(a) {
}

// @ngInject
// already has .$inject array (after function definition)
function Ctrl2(a) {
}
Ctrl2.$inject = ["serviceName"];

function outer() {
    MyCtrl["$inject"] = ["asdf"];
    return {
        controller: MyCtrl,
    };

    // @ngInject
    function MyCtrl(a) {
    }
}
