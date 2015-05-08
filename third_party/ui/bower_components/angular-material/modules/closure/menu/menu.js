/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.6.0
 */
goog.provide('ng.material.components.menu');


angular.module('material.components.menu', [
])

.factory('$mdMenu', MenuProvider);

function MenuProvider($$interimElementProvider) {
  return $$interimElementProvider('$mdMenu')
    .setDefaults({
      methods: ['placement'],
      options: menuDefaultOptions
    });

  /* @ngInject */
  function menuDefaultOptions() {

  }
}
MenuProvider.$inject = ["$$interimElementProvider"];


