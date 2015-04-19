function BindingsCtrl($scope) {
  $scope.planets = [
    { name: 'Mercury',
      radius: 1516
    },
    { name: 'Venus',
      radius: 3760
    },
    { name: 'Earth',
      radius: 3959,
      moons: ['Luna']
    },
    { name: 'Mars',
      radius: 2106,
      moons: ['Phobos', 'Deimos']
    },
    { name: 'Jupiter',
      radius: 43441,
      moons: ['Europa', 'Io', 'Ganymede', 'Castillo']
    },
    { name: 'Saturn',
      radius: 36184,
      moons: ['Titan', 'Rhea', 'Iapetus', 'Dione']
    },
    { name: 'Uranus',
      radius: 15759,
      moons: ['Titania', 'Oberon', 'Umbriel', 'Ariel']
    },
    { name: 'Neptune',
      radius: 15299,
      moons: ['Triton', 'Proteus', 'Nereid', 'Larissa']
    }
  ];

  $scope.planet = $scope.planets[0];

  $scope.getRadiusKm = function() {
    return $scope.planet.radius * 0.6213;
  };
}

BindingsCtrl.$inject = ['$scope'];
