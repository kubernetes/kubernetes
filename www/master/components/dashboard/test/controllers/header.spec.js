'use strict';

describe('header controller', function() {
    beforeEach(module('kubernetesApp.components.dashboard'));

    beforeEach(inject(function($rootScope, $location, $controller) {
        this.rootScope = $rootScope;
        this.scope = $rootScope.$new();

        this.location = $location;
        spyOn(this.location, 'path');

        this.controller = $controller;
        this.ctrl = this.controller('HeaderCtrl', {
            $scope: this.scope
        });
        this.scope.$apply();
    }));

    describe('subPages', function() {
        it('is defined', function() {
            expect(this.scope.subPages).not.toBeUndefined();
        });

        it('is an array', function() {
            expect(Array.isArray(this.scope.subPages)).toBeTruthy();
        });

        it('is not empty', function() {
            expect(this.scope.subPages.length).toBeGreaterThan(0);
        });

        describe('each subPage', function() {
            it('has a category', function() {
                this.scope.subPages.forEach(function(subPage) {
                    expect(subPage.category).toBeTruthy();
                });
            });

            it('has a name', function() {
                this.scope.subPages.forEach(function(subPage) {
                    expect(subPage.name).toBeTruthy();
                });
            });

            it('has a value', function() {
                this.scope.subPages.forEach(function(subPage) {
                    expect(subPage.value).toBeTruthy();
                });
            });
        });
    });

    describe('Pages', function() {
        it('does not change location on first detected change', function() {
            expect(this.location.path).not.toHaveBeenCalled();
        });

        it('changes location on second detected change', function() {
            var _this = this;
            this.scope.$apply(function() {
                _this.scope.Pages = 'test_Pages';
            });
            expect(this.location.path).toHaveBeenCalledWith('test_Pages');
        });
    });
});
