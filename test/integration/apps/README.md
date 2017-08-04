## What this does this do?
This is a Bottom-Up integration tests suite for apps, this test suite has 5 stages.  Each stage integrates multiple component and verifies if they work well togather.  
Lowest layer verifies the core components and the later layers go on and verify more complex components.

| S.NO.  |  Layer Name         | Components                                  | Remarks              |
|:------:| :---------------:   | :---------------:                           | :----------:               |
|   1    | Core                | pod/ node / scheduler | Basic sanity tests  |  Basic Sanity testing by integrating core components
|   2    | Configuration       | Affinity / Service  / Volume / Disruption   |  Check if pod works with its closest configurable components
|   3    | Pod Controller      | RS  /  StatefulSet / Jobs / Daemon          |  Check if RS / SS and jobs manage pods as expected
|   4    | Higher Controllers  | Deployment / Cornjobs                       |  Check if deployments and Cronjobs create RS and jobs 
|   5    | Upgrades            | Upgrade tests on all the three controllers  |  Check if upgrade and history management works across all the controlers

## How to run 
TODO

## How to add tests
TODO
