function ShadowCtrl($scope) {
  // This is terrible Angular.js style, do not put DOM manipulation inside
  // controllers like this.
  var parentShadowRoot = document.querySelector('#innerDiv')
      .createShadowRoot();
  parentShadowRoot.appendChild(document.querySelector('#parentTemplate')
      .content.cloneNode(true));
  var olderShadowRoot = parentShadowRoot.querySelector("#parentDiv")
      .createShadowRoot();
  olderShadowRoot.appendChild(document.querySelector('#olderChildTemplate')
      .content.cloneNode(true));
  var youngerShadowRoot = parentShadowRoot.querySelector("#parentDiv")
      .createShadowRoot();
  youngerShadowRoot.appendChild(document.querySelector('#youngerChildTemplate')
      .content.cloneNode(true));
}

RepeaterCtrl.$inject = ['$scope'];
