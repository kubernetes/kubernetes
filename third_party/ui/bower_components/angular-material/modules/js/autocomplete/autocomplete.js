/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.8.1
 */
(function () {
  'use strict';
  /**
   * @ngdoc module
   * @name material.components.autocomplete
   */
  /*
   * @see js folder for autocomplete implementation
   */
  angular.module('material.components.autocomplete', [
    'material.core',
    'material.components.icon'
  ]);
})();

(function () {
  'use strict';
  angular
      .module('material.components.autocomplete')
      .controller('MdAutocompleteCtrl', MdAutocompleteCtrl);

  function MdAutocompleteCtrl ($scope, $element, $timeout, $q, $mdUtil, $mdConstant) {

    //-- private variables
    var self = this,
        itemParts = $scope.itemsExpr.split(/ in /i),
        itemExpr = itemParts[1],
        elements = {
          main:  $element[0],
          ul:    $element[0].getElementsByTagName('ul')[0],
          input: $element[0].getElementsByTagName('input')[0]
        },
        promise = null,
        cache = {};

    //-- public variables
    self.scope = $scope;
    self.parent = $scope.$parent;
    self.itemName = itemParts[0];
    self.matches = [];
    self.loading = false;
    self.hidden = true;
    self.index = 0;
    self.keydown = keydown;
    self.blur = blur;
    self.clear = clearValue;
    self.select = select;
    self.getCurrentDisplayValue = getCurrentDisplayValue;
    self.fetch = $mdUtil.debounce(fetchResults);

    return init();

    //-- start method definitions
    function init () {
      configureWatchers();
      configureAria();
    }

    function configureAria () {
      var ul = angular.element(elements.ul),
          input = angular.element(elements.input),
          id = ul.attr('id') || 'ul_' + $mdUtil.nextUid();
      ul.attr('id', id);
      input.attr('aria-owns', id);
    }

    function getItemScope (item) {
      if (!item) return;
      var locals = {};
      if (self.itemName) locals[self.itemName] = $scope.selectedItem;
      return locals;
    }

    function configureWatchers () {
      $scope.$watch('searchText', function (searchText) {
        self.index = -1;
        if (!searchText || searchText.length < Math.max(parseInt($scope.minLength, 10), 1)) {
          self.loading = false;
          self.matches = [];
          self.hidden = shouldHide();
          return;
        }
        var term = searchText.toLowerCase();
        if (promise && promise.cancel) {
          promise.cancel();
          promise = null;
        }
        if (!$scope.noCache && cache[term]) {
          self.matches = cache[term];
        } else {
          self.fetch(searchText);
        }
        self.hidden = shouldHide();
        if ($scope.textChange) $scope.textChange(getItemScope($scope.selectedItem));
      });
      $scope.$watch('selectedItem', function (selectedItem) {
        if ($scope.itemChange) $scope.itemChange(getItemScope(selectedItem));
      });
    }

    function fetchResults (searchText) {
      var items = $scope.$parent.$eval(itemExpr),
          term = searchText.toLowerCase();
      if (angular.isArray(items)) {
        handleResults(items);
      } else {
        self.loading = true;
        promise = $q.when(items).then(handleResults);
      }
      function handleResults (matches) {
        cache[term] = matches;
        if (searchText !== $scope.searchText) return; //-- just cache the results if old request
        promise = null;
        self.loading = false;
        self.matches = matches;
        self.hidden = shouldHide();
      }
    }

    function blur () {
      self.hidden = true;
    }

    function keydown (event) {
      switch (event.keyCode) {
        case $mdConstant.KEY_CODE.DOWN_ARROW:
            if (self.loading) return;
            event.preventDefault();
            self.index = Math.min(self.index + 1, self.matches.length - 1);
            updateScroll();
            break;
        case $mdConstant.KEY_CODE.UP_ARROW:
            if (self.loading) return;
            event.preventDefault();
            self.index = Math.max(0, self.index - 1);
            updateScroll();
            break;
        case $mdConstant.KEY_CODE.ENTER:
            if (self.loading || self.index < 0) return;
            event.preventDefault();
            select(self.index);
            break;
        case $mdConstant.KEY_CODE.ESCAPE:
            self.matches = [];
            self.hidden = true;
            self.index = -1;
            break;
        case $mdConstant.KEY_CODE.TAB:
            break;
        default:
      }
    }

    function clearValue () {
      $scope.searchText = '';
      select(-1);
      elements.input.focus();
    }

    function shouldHide () {
      return self.matches.length === 1 && $scope.searchText === getDisplayValue(self.matches[0]);
    }

    function getCurrentDisplayValue () {
      return getDisplayValue(self.matches[self.index]);
    }

    function getDisplayValue (item) {
      return (item && $scope.itemText) ? $scope.itemText(getItemScope(item)) : item;
    }

    function select (index) {
      $scope.selectedItem = self.matches[index];
      $scope.searchText = getDisplayValue($scope.selectedItem) || $scope.searchText;
      self.hidden = true;
      self.index = -1;
      self.matches = [];
    }

    function updateScroll () {
      var top = 41 * self.index,
          bot = top + 41,
          hgt = 41 * 5.5;
      if (top < elements.ul.scrollTop) {
        elements.ul.scrollTop = top;
      } else if (bot > elements.ul.scrollTop + hgt) {
        elements.ul.scrollTop = bot - hgt;
      }
    }

  }
  MdAutocompleteCtrl.$inject = ["$scope", "$element", "$timeout", "$q", "$mdUtil", "$mdConstant"];
})();

(function () {
  'use strict';
  angular
      .module('material.components.autocomplete')
      .directive('mdAutocomplete', MdAutocomplete);

  /**
   * @ngdoc directive
   * @name mdAutocomplete
   * @module material.components.autocomplete
   *
   * @description
   * `<md-autocomplete>` is a special input component with a drop-down of all possible matches to a custom query.
   * This component allows you to provide real-time suggestions as the user types in the input area.
   *
   * @param {string=} md-search-text A model to bind the search query text to
   * @param {object=} md-selected-item A model to bind the selected item to
   * @param {expression} md-items An expression in the format of `item in items` to iterate over matches for your search.
   * @param {string=} md-item-text An expression that will convert your object to a single string.
   * @param {string=} placeholder Placeholder text that will be forwarded to the input.
   * @param {boolean=} md-no-cache Disables the internal caching that happens in autocomplete
   * @param {expression} md-selected-item-change An expression to be run each time a new item is selected
   * @param {expression} md-search-text-change An expression to be run each time the search text updates
   * @param {boolean=} ng-disabled Determines whether or not to disable the input field
   * @param {number=} md-min-length Specifies the minimum length of text before autocomplete will make suggestions
   *
   * @usage
   * <hljs lang="html">
   *   <md-autocomplete
   *       md-selected-item="selectedItem"
   *       md-search-text="searchText"
   *       md-items="item in getMatches(searchText)"
   *       md-item-text="item.display">
   *     <span md-highlight-text="searchText">{{item.display}}</span>
   *   </md-autocomplete>
   * </hljs>
   */

  function MdAutocomplete () {
    return {
      template:     '\
        <md-autocomplete-wrap role="listbox">\
          <input type="text"\
              ng-disabled="isDisabled"\
              ng-model="searchText"\
              ng-keydown="$mdAutocompleteCtrl.keydown($event)"\
              ng-blur="$mdAutocompleteCtrl.blur()"\
              placeholder="{{placeholder}}"\
              aria-label="{{placeholder}}"\
              aria-autocomplete="list"\
              aria-haspopup="true"\
              aria-activedescendant=""\
              aria-expanded="{{!$mdAutocompleteCtrl.hidden}}"/>\
          <button\
              type="button"\
              ng-if="searchText"\
              ng-click="$mdAutocompleteCtrl.clear()">\
              <md-icon md-svg-icon="cancel"></md-icon>\
              <span class="visually-hidden">Clear</span>\
              </button>\
          <md-progress-linear ng-if="$mdAutocompleteCtrl.loading" md-mode="indeterminate"></md-progress-linear>\
        </md-autocomplete-wrap>\
        <ul role="presentation">\
          <li ng-repeat="(index, item) in $mdAutocompleteCtrl.matches"\
              ng-class="{ selected: index === $mdAutocompleteCtrl.index }"\
              ng-show="searchText && !$mdAutocompleteCtrl.hidden"\
              ng-click="$mdAutocompleteCtrl.select(index)"\
              ng-transclude\
              md-autocomplete-list-item="$mdAutocompleteCtrl.itemName">\
          </li>\
        </ul>\
        <aria-status\
            class="visually-hidden"\
            role="status"\
            aria-live="assertive">\
          <p ng-if="$mdAutocompleteCtrl.index === -1 && $mdAutocompleteCtrl.matches.length === 1">There is 1 match available.</p>\
          <p ng-if="$mdAutocompleteCtrl.index === -1 && $mdAutocompleteCtrl.matches.length > 1">There are {{$mdAutocompleteCtrl.matches.length}} matches available.</p>\
          <p ng-if="$mdAutocompleteCtrl.index >= 0">{{ $mdAutocompleteCtrl.getCurrentDisplayValue() }}</p>\
        </aria-status>',
      transclude:   true,
      controller:   'MdAutocompleteCtrl',
      controllerAs: '$mdAutocompleteCtrl',
      scope:        {
        searchText:   '=mdSearchText',
        selectedItem: '=mdSelectedItem',
        itemsExpr:    '@mdItems',
        itemText:     '&mdItemText',
        placeholder:  '@placeholder',
        noCache:      '=mdNoCache',
        itemChange:   '&mdSelectedItemChange',
        textChange:   '&mdSearchTextChange',
        isDisabled:   '=ngDisabled',
        minLength:    '=mdMinLength'
      }
    };
  }
})();

(function () {
  'use strict';
  angular
      .module('material.components.autocomplete')
      .controller('MdHighlightCtrl', MdHighlightCtrl);

  function MdHighlightCtrl ($scope, $element, $interpolate) {
    var term = $element.attr('md-highlight-text'),
        text = $interpolate($element.text())($scope);
    $scope.$watch(term, function (term) {
      var regex = new RegExp('^' + sanitize(term), 'i'),
          html = text.replace(regex, '<span class="highlight">$&</span>');
      $element.html(html);
    });

    function sanitize (term) {
      if (!term) return term;
      return term.replace(/[\*\[\]\(\)\{\}\\\^\$]/g, '\\$&');
    }
  }
  MdHighlightCtrl.$inject = ["$scope", "$element", "$interpolate"];

})();

(function () {
  'use strict';
  angular
      .module('material.components.autocomplete')
      .directive('mdHighlightText', MdHighlight);

  /**
   * @ngdoc directive
   * @name mdHighlightText
   * @module material.components.autocomplete
   *
   * @description
   * The `md-highlight-text` directive allows you to specify text that should be highlighted within
   * an element.  Highlighted text will be wrapped in `<span class="highlight"></span>` which can
   * be styled through CSS.  Please note that child elements may not be used with this directive.
   *
   * @param {string=} md-highlight-text A model to be searched for
   *
   * @usage
   * <hljs lang="html">
   * <input placeholder="Enter a search term..." ng-model="searchTerm" type="text" />
   * <ul>
   *   <li ng-repeat="result in results" md-highlight-text="searchTerm">
   *     {{result.text}}
   *   </li>
   * </ul>
   * </hljs>
   */

  function MdHighlight () {
    return {
      terminal: true,
      scope: false,
      controller: 'MdHighlightCtrl'
    };
  }
})();

(function () {
  'use strict';
  angular
      .module('material.components.autocomplete')
      .directive('mdAutocompleteListItem', MdAutocompleteListItem);

  function MdAutocompleteListItem ($compile, $mdUtil) {
    return {
      require: '^?mdAutocomplete',
      terminal: true,
      link: link,
      scope: false
    };
    function link (scope, element, attr, ctrl) {
      var newScope = ctrl.parent.$new(false, ctrl.parent),
          itemName = ctrl.scope.$eval(attr.mdAutocompleteListItem);
      newScope[itemName] = scope.item;
      $compile(element.contents())(newScope);
      element.attr({ 'role': 'option', 'id': 'item_' + $mdUtil.nextUid() });
    }
  }
  MdAutocompleteListItem.$inject = ["$compile", "$mdUtil"];
})();
