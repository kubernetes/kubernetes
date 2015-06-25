/**=========================================================
 * Module: home-page.js
 * Page Controller
 =========================================================*/

app.controller('PageCtrl', [
  '$scope',
  '$timeout',
  '$mdSidenav',
  'menu',
  '$rootScope',
  function($scope, $timeout, $mdSidenav, menu, $rootScope) {
  $scope.menu = menu;

  $scope.path = path;
  $scope.goHome = goHome;
  $scope.openMenu = openMenu;
  $rootScope.openMenu = openMenu;
  $scope.closeMenu = closeMenu;
  $scope.isSectionSelected = isSectionSelected;

  $rootScope.$on('$locationChangeSuccess', openPage);

  // Methods used by menuLink and menuToggle directives
  this.isOpen = isOpen;
  this.isSelected = isSelected;
  this.toggleOpen = toggleOpen;
  this.shouldLockOpen = shouldLockOpen;
  $scope.toggleKubernetesUiMenu = toggleKubernetesUiMenu;

  var mainContentArea = document.querySelector("[role='main']");
  var kubernetesUiMenu = document.querySelector("[role='kubernetes-ui-menu']");

  // *********************
  // Internal methods
  // *********************

  var _t = false;

  $scope.showKubernetesUiMenu = false;

  function shouldLockOpen() {
    return _t;
  }

  function toggleKubernetesUiMenu() {
    $scope.showKubernetesUiMenu = !$scope.showKubernetesUiMenu;
  }

  function closeMenu() {
    $timeout(function() {
      $mdSidenav('left').close();
    });
  }

  function openMenu() {
    $timeout(function() {
      _t = !$mdSidenav('left').isOpen();
      $mdSidenav('left').toggle();
    });
  }

  function path() {
    return $location.path();
  }

  function goHome($event) {
    menu.selectPage(null, null);
    $location.path( '/' );
  }

  function openPage() {
    $scope.closeMenu();
    mainContentArea.focus();
  }

  function isSelected(page) {
    return menu.isPageSelected(page);
  }

  function isSectionSelected(section) {
    var selected = false;
    var openedSection = menu.openedSection;
    if(openedSection === section){
      selected = true;
    }
    else if(section.children) {
      section.children.forEach(function(childSection) {
        if(childSection === openedSection){
          selected = true;
        }
      });
    }
    return selected;
  }

  function isOpen(section) {
    return menu.isSectionSelected(section);
  }

  function toggleOpen(section) {
    menu.toggleSelectSection(section);
  }

  }
]).filter('humanizeDoc', function() {
  return function(doc) {
    if (!doc) return;
    if (doc.type === 'directive') {
      return doc.name.replace(/([A-Z])/g, function($1) {
        return '-'+$1.toLowerCase();
      });
    }
    return doc.label || doc.name;
  }; });
