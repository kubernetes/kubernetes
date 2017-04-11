module.exports = function(hljs) {
  var CPP = hljs.getLanguage('cpp').exports;

  // In SQF, a variable start with _
  var VARIABLE = {
    className: 'variable',
    begin: /\b_+[a-zA-Z_]\w*/
  };

  // In SQF, a function should fit myTag_fnc_myFunction pattern
  // https://community.bistudio.com/wiki/Functions_Library_(Arma_3)#Adding_a_Function
  var FUNCTION = {
    className: 'title',
    begin: /[a-zA-Z][a-zA-Z0-9]+_fnc_\w*/
  };

  // In SQF strings, quotes matching the start are escaped by adding a consecutive.
  // Example of single escaped quotes: " "" " and  ' '' '.
  var STRINGS = {
    className: 'string',
    variants: [
      {
        begin: '"',
        end: '"',
        contains: [{begin: '""', relevance: 0}]
      },
      {
        begin: '\'',
        end: '\'',
        contains: [{begin: '\'\'', relevance: 0}]
      }
    ]
  };

  return {
    aliases: ['sqf'],
    case_insensitive: true,
    keywords: {
      keyword:
        'case catch default do else exit exitWith for forEach from if ' +
        'switch then throw to try waitUntil while with',
      built_in:
        'abs accTime acos action actionIDs actionKeys actionKeysImages actionKeysNames ' +
        'actionKeysNamesArray actionName actionParams activateAddons activatedAddons activateKey ' +
        'add3DENConnection add3DENEventHandler add3DENLayer addAction addBackpack addBackpackCargo ' +
        'addBackpackCargoGlobal addBackpackGlobal addCamShake addCuratorAddons addCuratorCameraArea ' +
        'addCuratorEditableObjects addCuratorEditingArea addCuratorPoints addEditorObject addEventHandler ' +
        'addGoggles addGroupIcon addHandgunItem addHeadgear addItem addItemCargo addItemCargoGlobal ' +
        'addItemPool addItemToBackpack addItemToUniform addItemToVest addLiveStats addMagazine ' +
        'addMagazineAmmoCargo addMagazineCargo addMagazineCargoGlobal addMagazineGlobal addMagazinePool ' +
        'addMagazines addMagazineTurret addMenu addMenuItem addMissionEventHandler addMPEventHandler ' +
        'addMusicEventHandler addOwnedMine addPlayerScores addPrimaryWeaponItem ' +
        'addPublicVariableEventHandler addRating addResources addScore addScoreSide addSecondaryWeaponItem ' +
        'addSwitchableUnit addTeamMember addToRemainsCollector addUniform addVehicle addVest addWaypoint ' +
        'addWeapon addWeaponCargo addWeaponCargoGlobal addWeaponGlobal addWeaponItem addWeaponPool ' +
        'addWeaponTurret agent agents AGLToASL aimedAtTarget aimPos airDensityRTD airportSide ' +
        'AISFinishHeal alive all3DENEntities allControls allCurators allCutLayers allDead allDeadMen ' +
        'allDisplays allGroups allMapMarkers allMines allMissionObjects allow3DMode allowCrewInImmobile ' +
        'allowCuratorLogicIgnoreAreas allowDamage allowDammage allowFileOperations allowFleeing allowGetIn ' +
        'allowSprint allPlayers allSites allTurrets allUnits allUnitsUAV allVariables ammo and animate ' +
        'animateDoor animateSource animationNames animationPhase animationSourcePhase animationState ' +
        'append apply armoryPoints arrayIntersect asin ASLToAGL ASLToATL assert assignAsCargo ' +
        'assignAsCargoIndex assignAsCommander assignAsDriver assignAsGunner assignAsTurret assignCurator ' +
        'assignedCargo assignedCommander assignedDriver assignedGunner assignedItems assignedTarget ' +
        'assignedTeam assignedVehicle assignedVehicleRole assignItem assignTeam assignToAirport atan atan2 ' +
        'atg ATLToASL attachedObject attachedObjects attachedTo attachObject attachTo attackEnabled ' +
        'backpack backpackCargo backpackContainer backpackItems backpackMagazines backpackSpaceFor ' +
        'behaviour benchmark binocular blufor boundingBox boundingBoxReal boundingCenter breakOut breakTo ' +
        'briefingName buildingExit buildingPos buttonAction buttonSetAction cadetMode call callExtension ' +
        'camCommand camCommit camCommitPrepared camCommitted camConstuctionSetParams camCreate camDestroy ' +
        'cameraEffect cameraEffectEnableHUD cameraInterest cameraOn cameraView campaignConfigFile ' +
        'camPreload camPreloaded camPrepareBank camPrepareDir camPrepareDive camPrepareFocus camPrepareFov ' +
        'camPrepareFovRange camPreparePos camPrepareRelPos camPrepareTarget camSetBank camSetDir ' +
        'camSetDive camSetFocus camSetFov camSetFovRange camSetPos camSetRelPos camSetTarget camTarget ' +
        'camUseNVG canAdd canAddItemToBackpack canAddItemToUniform canAddItemToVest ' +
        'cancelSimpleTaskDestination canFire canMove canSlingLoad canStand canSuspend canUnloadInCombat ' +
        'canVehicleCargo captive captiveNum cbChecked cbSetChecked ceil channelEnabled cheatsEnabled ' +
        'checkAIFeature checkVisibility civilian className clearAllItemsFromBackpack clearBackpackCargo ' +
        'clearBackpackCargoGlobal clearGroupIcons clearItemCargo clearItemCargoGlobal clearItemPool ' +
        'clearMagazineCargo clearMagazineCargoGlobal clearMagazinePool clearOverlay clearRadio ' +
        'clearWeaponCargo clearWeaponCargoGlobal clearWeaponPool clientOwner closeDialog closeDisplay ' +
        'closeOverlay collapseObjectTree collect3DENHistory combatMode commandArtilleryFire commandChat ' +
        'commander commandFire commandFollow commandFSM commandGetOut commandingMenu commandMove ' +
        'commandRadio commandStop commandSuppressiveFire commandTarget commandWatch comment commitOverlay ' +
        'compile compileFinal completedFSM composeText configClasses configFile configHierarchy configName ' +
        'configNull configProperties configSourceAddonList configSourceMod configSourceModList ' +
        'connectTerminalToUAV controlNull controlsGroupCtrl copyFromClipboard copyToClipboard ' +
        'copyWaypoints cos count countEnemy countFriendly countSide countType countUnknown ' +
        'create3DENComposition create3DENEntity createAgent createCenter createDialog createDiaryLink ' +
        'createDiaryRecord createDiarySubject createDisplay createGearDialog createGroup ' +
        'createGuardedPoint createLocation createMarker createMarkerLocal createMenu createMine ' +
        'createMissionDisplay createMPCampaignDisplay createSimpleObject createSimpleTask createSite ' +
        'createSoundSource createTask createTeam createTrigger createUnit createVehicle createVehicleCrew ' +
        'createVehicleLocal crew ctrlActivate ctrlAddEventHandler ctrlAngle ctrlAutoScrollDelay ' +
        'ctrlAutoScrollRewind ctrlAutoScrollSpeed ctrlChecked ctrlClassName ctrlCommit ctrlCommitted ' +
        'ctrlCreate ctrlDelete ctrlEnable ctrlEnabled ctrlFade ctrlHTMLLoaded ctrlIDC ctrlIDD ' +
        'ctrlMapAnimAdd ctrlMapAnimClear ctrlMapAnimCommit ctrlMapAnimDone ctrlMapCursor ctrlMapMouseOver ' +
        'ctrlMapScale ctrlMapScreenToWorld ctrlMapWorldToScreen ctrlModel ctrlModelDirAndUp ctrlModelScale ' +
        'ctrlParent ctrlParentControlsGroup ctrlPosition ctrlRemoveAllEventHandlers ctrlRemoveEventHandler ' +
        'ctrlScale ctrlSetActiveColor ctrlSetAngle ctrlSetAutoScrollDelay ctrlSetAutoScrollRewind ' +
        'ctrlSetAutoScrollSpeed ctrlSetBackgroundColor ctrlSetChecked ctrlSetEventHandler ctrlSetFade ' +
        'ctrlSetFocus ctrlSetFont ctrlSetFontH1 ctrlSetFontH1B ctrlSetFontH2 ctrlSetFontH2B ctrlSetFontH3 ' +
        'ctrlSetFontH3B ctrlSetFontH4 ctrlSetFontH4B ctrlSetFontH5 ctrlSetFontH5B ctrlSetFontH6 ' +
        'ctrlSetFontH6B ctrlSetFontHeight ctrlSetFontHeightH1 ctrlSetFontHeightH2 ctrlSetFontHeightH3 ' +
        'ctrlSetFontHeightH4 ctrlSetFontHeightH5 ctrlSetFontHeightH6 ctrlSetFontHeightSecondary ' +
        'ctrlSetFontP ctrlSetFontPB ctrlSetFontSecondary ctrlSetForegroundColor ctrlSetModel ' +
        'ctrlSetModelDirAndUp ctrlSetModelScale ctrlSetPosition ctrlSetScale ctrlSetStructuredText ' +
        'ctrlSetText ctrlSetTextColor ctrlSetTooltip ctrlSetTooltipColorBox ctrlSetTooltipColorShade ' +
        'ctrlSetTooltipColorText ctrlShow ctrlShown ctrlText ctrlTextHeight ctrlType ctrlVisible ' +
        'curatorAddons curatorCamera curatorCameraArea curatorCameraAreaCeiling curatorCoef ' +
        'curatorEditableObjects curatorEditingArea curatorEditingAreaType curatorMouseOver curatorPoints ' +
        'curatorRegisteredObjects curatorSelected curatorWaypointCost current3DENOperation currentChannel ' +
        'currentCommand currentMagazine currentMagazineDetail currentMagazineDetailTurret ' +
        'currentMagazineTurret currentMuzzle currentNamespace currentTask currentTasks currentThrowable ' +
        'currentVisionMode currentWaypoint currentWeapon currentWeaponMode currentWeaponTurret ' +
        'currentZeroing cursorObject cursorTarget customChat customRadio cutFadeOut cutObj cutRsc cutText ' +
        'damage date dateToNumber daytime deActivateKey debriefingText debugFSM debugLog deg ' +
        'delete3DENEntities deleteAt deleteCenter deleteCollection deleteEditorObject deleteGroup ' +
        'deleteIdentity deleteLocation deleteMarker deleteMarkerLocal deleteRange deleteResources ' +
        'deleteSite deleteStatus deleteTeam deleteVehicle deleteVehicleCrew deleteWaypoint detach ' +
        'detectedMines diag_activeMissionFSMs diag_activeScripts diag_activeSQFScripts ' +
        'diag_activeSQSScripts diag_captureFrame diag_captureSlowFrame diag_codePerformance diag_drawMode ' +
        'diag_enable diag_enabled diag_fps diag_fpsMin diag_frameNo diag_list diag_log diag_logSlowFrame ' +
        'diag_mergeConfigFile diag_recordTurretLimits diag_tickTime diag_toggle dialog diarySubjectExists ' +
        'didJIP didJIPOwner difficulty difficultyEnabled difficultyEnabledRTD difficultyOption direction ' +
        'directSay disableAI disableCollisionWith disableConversation disableDebriefingStats ' +
        'disableNVGEquipment disableRemoteSensors disableSerialization disableTIEquipment ' +
        'disableUAVConnectability disableUserInput displayAddEventHandler displayCtrl displayNull ' +
        'displayParent displayRemoveAllEventHandlers displayRemoveEventHandler displaySetEventHandler ' +
        'dissolveTeam distance distance2D distanceSqr distributionRegion do3DENAction doArtilleryFire ' +
        'doFire doFollow doFSM doGetOut doMove doorPhase doStop doSuppressiveFire doTarget doWatch ' +
        'drawArrow drawEllipse drawIcon drawIcon3D drawLine drawLine3D drawLink drawLocation drawPolygon ' +
        'drawRectangle driver drop east echo edit3DENMissionAttributes editObject editorSetEventHandler ' +
        'effectiveCommander emptyPositions enableAI enableAIFeature enableAimPrecision enableAttack ' +
        'enableAudioFeature enableCamShake enableCaustics enableChannel enableCollisionWith enableCopilot ' +
        'enableDebriefingStats enableDiagLegend enableEndDialog enableEngineArtillery enableEnvironment ' +
        'enableFatigue enableGunLights enableIRLasers enableMimics enablePersonTurret enableRadio ' +
        'enableReload enableRopeAttach enableSatNormalOnDetail enableSaving enableSentences ' +
        'enableSimulation enableSimulationGlobal enableStamina enableTeamSwitch enableUAVConnectability ' +
        'enableUAVWaypoints enableVehicleCargo endLoadingScreen endMission engineOn enginesIsOnRTD ' +
        'enginesRpmRTD enginesTorqueRTD entities estimatedEndServerTime estimatedTimeLeft ' +
        'evalObjectArgument everyBackpack everyContainer exec execEditorScript execFSM execVM exp ' +
        'expectedDestination exportJIPMessages eyeDirection eyePos face faction fadeMusic fadeRadio ' +
        'fadeSound fadeSpeech failMission fillWeaponsFromPool find findCover findDisplay findEditorObject ' +
        'findEmptyPosition findEmptyPositionReady findNearestEnemy finishMissionInit finite fire ' +
        'fireAtTarget firstBackpack flag flagOwner flagSide flagTexture fleeing floor flyInHeight ' +
        'flyInHeightASL fog fogForecast fogParams forceAddUniform forcedMap forceEnd forceMap forceRespawn ' +
        'forceSpeed forceWalk forceWeaponFire forceWeatherChange forEachMember forEachMemberAgent ' +
        'forEachMemberTeam format formation formationDirection formationLeader formationMembers ' +
        'formationPosition formationTask formatText formLeader freeLook fromEditor fuel fullCrew ' +
        'gearIDCAmmoCount gearSlotAmmoCount gearSlotData get3DENActionState get3DENAttribute get3DENCamera ' +
        'get3DENConnections get3DENEntity get3DENEntityID get3DENGrid get3DENIconsVisible ' +
        'get3DENLayerEntities get3DENLinesVisible get3DENMissionAttribute get3DENMouseOver get3DENSelected ' +
        'getAimingCoef getAllHitPointsDamage getAllOwnedMines getAmmoCargo getAnimAimPrecision ' +
        'getAnimSpeedCoef getArray getArtilleryAmmo getArtilleryComputerSettings getArtilleryETA ' +
        'getAssignedCuratorLogic getAssignedCuratorUnit getBackpackCargo getBleedingRemaining ' +
        'getBurningValue getCameraViewDirection getCargoIndex getCenterOfMass getClientState ' +
        'getClientStateNumber getConnectedUAV getCustomAimingCoef getDammage getDescription getDir ' +
        'getDirVisual getDLCs getEditorCamera getEditorMode getEditorObjectScope getElevationOffset ' +
        'getFatigue getFriend getFSMVariable getFuelCargo getGroupIcon getGroupIconParams getGroupIcons ' +
        'getHideFrom getHit getHitIndex getHitPointDamage getItemCargo getMagazineCargo getMarkerColor ' +
        'getMarkerPos getMarkerSize getMarkerType getMass getMissionConfig getMissionConfigValue ' +
        'getMissionDLCs getMissionLayerEntities getModelInfo getMousePosition getNumber getObjectArgument ' +
        'getObjectChildren getObjectDLC getObjectMaterials getObjectProxy getObjectTextures getObjectType ' +
        'getObjectViewDistance getOxygenRemaining getPersonUsedDLCs getPilotCameraDirection ' +
        'getPilotCameraPosition getPilotCameraRotation getPilotCameraTarget getPlayerChannel ' +
        'getPlayerScores getPlayerUID getPos getPosASL getPosASLVisual getPosASLW getPosATL ' +
        'getPosATLVisual getPosVisual getPosWorld getRelDir getRelPos getRemoteSensorsDisabled ' +
        'getRepairCargo getResolution getShadowDistance getShotParents getSlingLoad getSpeed getStamina ' +
        'getStatValue getSuppression getTerrainHeightASL getText getUnitLoadout getUnitTrait getVariable ' +
        'getVehicleCargo getWeaponCargo getWeaponSway getWPPos glanceAt globalChat globalRadio goggles ' +
        'goto group groupChat groupFromNetId groupIconSelectable groupIconsVisible groupId groupOwner ' +
        'groupRadio groupSelectedUnits groupSelectUnit grpNull gunner gusts halt handgunItems ' +
        'handgunMagazine handgunWeapon handsHit hasInterface hasPilotCamera hasWeapon hcAllGroups ' +
        'hcGroupParams hcLeader hcRemoveAllGroups hcRemoveGroup hcSelected hcSelectGroup hcSetGroup ' +
        'hcShowBar hcShownBar headgear hideBody hideObject hideObjectGlobal hideSelection hint hintC ' +
        'hintCadet hintSilent hmd hostMission htmlLoad HUDMovementLevels humidity image importAllGroups ' +
        'importance in inArea inAreaArray incapacitatedState independent inflame inflamed ' +
        'inGameUISetEventHandler inheritsFrom initAmbientLife inPolygon inputAction inRangeOfArtillery ' +
        'insertEditorObject intersect is3DEN is3DENMultiplayer isAbleToBreathe isAgent isArray ' +
        'isAutoHoverOn isAutonomous isAutotest isBleeding isBurning isClass isCollisionLightOn ' +
        'isCopilotEnabled isDedicated isDLCAvailable isEngineOn isEqualTo isEqualType isEqualTypeAll ' +
        'isEqualTypeAny isEqualTypeArray isEqualTypeParams isFilePatchingEnabled isFlashlightOn ' +
        'isFlatEmpty isForcedWalk isFormationLeader isHidden isInRemainsCollector ' +
        'isInstructorFigureEnabled isIRLaserOn isKeyActive isKindOf isLightOn isLocalized isManualFire ' +
        'isMarkedForCollection isMultiplayer isMultiplayerSolo isNil isNull isNumber isObjectHidden ' +
        'isObjectRTD isOnRoad isPipEnabled isPlayer isRealTime isRemoteExecuted isRemoteExecutedJIP ' +
        'isServer isShowing3DIcons isSprintAllowed isStaminaEnabled isSteamMission ' +
        'isStreamFriendlyUIEnabled isText isTouchingGround isTurnedOut isTutHintsEnabled isUAVConnectable ' +
        'isUAVConnected isUniformAllowed isVehicleCargo isWalking isWeaponDeployed isWeaponRested ' +
        'itemCargo items itemsWithMagazines join joinAs joinAsSilent joinSilent joinString kbAddDatabase ' +
        'kbAddDatabaseTargets kbAddTopic kbHasTopic kbReact kbRemoveTopic kbTell kbWasSaid keyImage ' +
        'keyName knowsAbout land landAt landResult language laserTarget lbAdd lbClear lbColor lbCurSel ' +
        'lbData lbDelete lbIsSelected lbPicture lbSelection lbSetColor lbSetCurSel lbSetData lbSetPicture ' +
        'lbSetPictureColor lbSetPictureColorDisabled lbSetPictureColorSelected lbSetSelectColor ' +
        'lbSetSelectColorRight lbSetSelected lbSetTooltip lbSetValue lbSize lbSort lbSortByValue lbText ' +
        'lbValue leader leaderboardDeInit leaderboardGetRows leaderboardInit leaveVehicle libraryCredits ' +
        'libraryDisclaimers lifeState lightAttachObject lightDetachObject lightIsOn lightnings limitSpeed ' +
        'linearConversion lineBreak lineIntersects lineIntersectsObjs lineIntersectsSurfaces ' +
        'lineIntersectsWith linkItem list listObjects ln lnbAddArray lnbAddColumn lnbAddRow lnbClear ' +
        'lnbColor lnbCurSelRow lnbData lnbDeleteColumn lnbDeleteRow lnbGetColumnsPosition lnbPicture ' +
        'lnbSetColor lnbSetColumnsPos lnbSetCurSelRow lnbSetData lnbSetPicture lnbSetText lnbSetValue ' +
        'lnbSize lnbText lnbValue load loadAbs loadBackpack loadFile loadGame loadIdentity loadMagazine ' +
        'loadOverlay loadStatus loadUniform loadVest local localize locationNull locationPosition lock ' +
        'lockCameraTo lockCargo lockDriver locked lockedCargo lockedDriver lockedTurret lockIdentity ' +
        'lockTurret lockWP log logEntities logNetwork logNetworkTerminate lookAt lookAtPos magazineCargo ' +
        'magazines magazinesAllTurrets magazinesAmmo magazinesAmmoCargo magazinesAmmoFull magazinesDetail ' +
        'magazinesDetailBackpack magazinesDetailUniform magazinesDetailVest magazinesTurret ' +
        'magazineTurretAmmo mapAnimAdd mapAnimClear mapAnimCommit mapAnimDone mapCenterOnCamera ' +
        'mapGridPosition markAsFinishedOnSteam markerAlpha markerBrush markerColor markerDir markerPos ' +
        'markerShape markerSize markerText markerType max members menuAction menuAdd menuChecked menuClear ' +
        'menuCollapse menuData menuDelete menuEnable menuEnabled menuExpand menuHover menuPicture ' +
        'menuSetAction menuSetCheck menuSetData menuSetPicture menuSetValue menuShortcut menuShortcutText ' +
        'menuSize menuSort menuText menuURL menuValue min mineActive mineDetectedBy missionConfigFile ' +
        'missionDifficulty missionName missionNamespace missionStart missionVersion mod modelToWorld ' +
        'modelToWorldVisual modParams moonIntensity moonPhase morale move move3DENCamera moveInAny ' +
        'moveInCargo moveInCommander moveInDriver moveInGunner moveInTurret moveObjectToEnd moveOut ' +
        'moveTime moveTo moveToCompleted moveToFailed musicVolume name nameSound nearEntities ' +
        'nearestBuilding nearestLocation nearestLocations nearestLocationWithDubbing nearestObject ' +
        'nearestObjects nearestTerrainObjects nearObjects nearObjectsReady nearRoads nearSupplies ' +
        'nearTargets needReload netId netObjNull newOverlay nextMenuItemIndex nextWeatherChange nMenuItems ' +
        'not numberToDate objectCurators objectFromNetId objectParent objNull objStatus onBriefingGroup ' +
        'onBriefingNotes onBriefingPlan onBriefingTeamSwitch onCommandModeChanged onDoubleClick ' +
        'onEachFrame onGroupIconClick onGroupIconOverEnter onGroupIconOverLeave onHCGroupSelectionChanged ' +
        'onMapSingleClick onPlayerConnected onPlayerDisconnected onPreloadFinished onPreloadStarted ' +
        'onShowNewObject onTeamSwitch openCuratorInterface openDLCPage openMap openYoutubeVideo opfor or ' +
        'orderGetIn overcast overcastForecast owner param params parseNumber parseText parsingNamespace ' +
        'particlesQuality pi pickWeaponPool pitch pixelGrid pixelGridBase pixelGridNoUIScale pixelH pixelW ' +
        'playableSlotsNumber playableUnits playAction playActionNow player playerRespawnTime playerSide ' +
        'playersNumber playGesture playMission playMove playMoveNow playMusic playScriptedMission ' +
        'playSound playSound3D position positionCameraToWorld posScreenToWorld posWorldToScreen ' +
        'ppEffectAdjust ppEffectCommit ppEffectCommitted ppEffectCreate ppEffectDestroy ppEffectEnable ' +
        'ppEffectEnabled ppEffectForceInNVG precision preloadCamera preloadObject preloadSound ' +
        'preloadTitleObj preloadTitleRsc preprocessFile preprocessFileLineNumbers primaryWeapon ' +
        'primaryWeaponItems primaryWeaponMagazine priority private processDiaryLink productVersion ' +
        'profileName profileNamespace profileNameSteam progressLoadingScreen progressPosition ' +
        'progressSetPosition publicVariable publicVariableClient publicVariableServer pushBack ' +
        'pushBackUnique putWeaponPool queryItemsPool queryMagazinePool queryWeaponPool rad radioChannelAdd ' +
        'radioChannelCreate radioChannelRemove radioChannelSetCallSign radioChannelSetLabel radioVolume ' +
        'rain rainbow random rank rankId rating rectangular registeredTasks registerTask reload ' +
        'reloadEnabled remoteControl remoteExec remoteExecCall remove3DENConnection remove3DENEventHandler ' +
        'remove3DENLayer removeAction removeAll3DENEventHandlers removeAllActions removeAllAssignedItems ' +
        'removeAllContainers removeAllCuratorAddons removeAllCuratorCameraAreas ' +
        'removeAllCuratorEditingAreas removeAllEventHandlers removeAllHandgunItems removeAllItems ' +
        'removeAllItemsWithMagazines removeAllMissionEventHandlers removeAllMPEventHandlers ' +
        'removeAllMusicEventHandlers removeAllOwnedMines removeAllPrimaryWeaponItems removeAllWeapons ' +
        'removeBackpack removeBackpackGlobal removeCuratorAddons removeCuratorCameraArea ' +
        'removeCuratorEditableObjects removeCuratorEditingArea removeDrawIcon removeDrawLinks ' +
        'removeEventHandler removeFromRemainsCollector removeGoggles removeGroupIcon removeHandgunItem ' +
        'removeHeadgear removeItem removeItemFromBackpack removeItemFromUniform removeItemFromVest ' +
        'removeItems removeMagazine removeMagazineGlobal removeMagazines removeMagazinesTurret ' +
        'removeMagazineTurret removeMenuItem removeMissionEventHandler removeMPEventHandler ' +
        'removeMusicEventHandler removeOwnedMine removePrimaryWeaponItem removeSecondaryWeaponItem ' +
        'removeSimpleTask removeSwitchableUnit removeTeamMember removeUniform removeVest removeWeapon ' +
        'removeWeaponGlobal removeWeaponTurret requiredVersion resetCamShake resetSubgroupDirection ' +
        'resistance resize resources respawnVehicle restartEditorCamera reveal revealMine reverse ' +
        'reversedMouseY roadAt roadsConnectedTo roleDescription ropeAttachedObjects ropeAttachedTo ' +
        'ropeAttachEnabled ropeAttachTo ropeCreate ropeCut ropeDestroy ropeDetach ropeEndPosition ' +
        'ropeLength ropes ropeUnwind ropeUnwound rotorsForcesRTD rotorsRpmRTD round runInitScript ' +
        'safeZoneH safeZoneW safeZoneWAbs safeZoneX safeZoneXAbs safeZoneY save3DENInventory saveGame ' +
        'saveIdentity saveJoysticks saveOverlay saveProfileNamespace saveStatus saveVar savingEnabled say ' +
        'say2D say3D scopeName score scoreSide screenshot screenToWorld scriptDone scriptName scriptNull ' +
        'scudState secondaryWeapon secondaryWeaponItems secondaryWeaponMagazine select selectBestPlaces ' +
        'selectDiarySubject selectedEditorObjects selectEditorObject selectionNames selectionPosition ' +
        'selectLeader selectMax selectMin selectNoPlayer selectPlayer selectRandom selectWeapon ' +
        'selectWeaponTurret sendAUMessage sendSimpleCommand sendTask sendTaskResult sendUDPMessage ' +
        'serverCommand serverCommandAvailable serverCommandExecutable serverName serverTime set ' +
        'set3DENAttribute set3DENAttributes set3DENGrid set3DENIconsVisible set3DENLayer ' +
        'set3DENLinesVisible set3DENMissionAttributes set3DENModelsVisible set3DENObjectType ' +
        'set3DENSelected setAccTime setAirportSide setAmmo setAmmoCargo setAnimSpeedCoef setAperture ' +
        'setApertureNew setArmoryPoints setAttributes setAutonomous setBehaviour setBleedingRemaining ' +
        'setCameraInterest setCamShakeDefParams setCamShakeParams setCamUseTi setCaptive setCenterOfMass ' +
        'setCollisionLight setCombatMode setCompassOscillation setCuratorCameraAreaCeiling setCuratorCoef ' +
        'setCuratorEditingAreaType setCuratorWaypointCost setCurrentChannel setCurrentTask ' +
        'setCurrentWaypoint setCustomAimCoef setDamage setDammage setDate setDebriefingText ' +
        'setDefaultCamera setDestination setDetailMapBlendPars setDir setDirection setDrawIcon ' +
        'setDropInterval setEditorMode setEditorObjectScope setEffectCondition setFace setFaceAnimation ' +
        'setFatigue setFlagOwner setFlagSide setFlagTexture setFog setFormation setFormationTask ' +
        'setFormDir setFriend setFromEditor setFSMVariable setFuel setFuelCargo setGroupIcon ' +
        'setGroupIconParams setGroupIconsSelectable setGroupIconsVisible setGroupId setGroupIdGlobal ' +
        'setGroupOwner setGusts setHideBehind setHit setHitIndex setHitPointDamage setHorizonParallaxCoef ' +
        'setHUDMovementLevels setIdentity setImportance setLeader setLightAmbient setLightAttenuation ' +
        'setLightBrightness setLightColor setLightDayLight setLightFlareMaxDistance setLightFlareSize ' +
        'setLightIntensity setLightnings setLightUseFlare setLocalWindParams setMagazineTurretAmmo ' +
        'setMarkerAlpha setMarkerAlphaLocal setMarkerBrush setMarkerBrushLocal setMarkerColor ' +
        'setMarkerColorLocal setMarkerDir setMarkerDirLocal setMarkerPos setMarkerPosLocal setMarkerShape ' +
        'setMarkerShapeLocal setMarkerSize setMarkerSizeLocal setMarkerText setMarkerTextLocal ' +
        'setMarkerType setMarkerTypeLocal setMass setMimic setMousePosition setMusicEffect ' +
        'setMusicEventHandler setName setNameSound setObjectArguments setObjectMaterial ' +
        'setObjectMaterialGlobal setObjectProxy setObjectTexture setObjectTextureGlobal ' +
        'setObjectViewDistance setOvercast setOwner setOxygenRemaining setParticleCircle setParticleClass ' +
        'setParticleFire setParticleParams setParticleRandom setPilotCameraDirection ' +
        'setPilotCameraRotation setPilotCameraTarget setPilotLight setPiPEffect setPitch setPlayable ' +
        'setPlayerRespawnTime setPos setPosASL setPosASL2 setPosASLW setPosATL setPosition setPosWorld ' +
        'setRadioMsg setRain setRainbow setRandomLip setRank setRectangular setRepairCargo ' +
        'setShadowDistance setShotParents setSide setSimpleTaskAlwaysVisible setSimpleTaskCustomData ' +
        'setSimpleTaskDescription setSimpleTaskDestination setSimpleTaskTarget setSimpleTaskType ' +
        'setSimulWeatherLayers setSize setSkill setSlingLoad setSoundEffect setSpeaker setSpeech ' +
        'setSpeedMode setStamina setStaminaScheme setStatValue setSuppression setSystemOfUnits ' +
        'setTargetAge setTaskResult setTaskState setTerrainGrid setText setTimeMultiplier setTitleEffect ' +
        'setTriggerActivation setTriggerArea setTriggerStatements setTriggerText setTriggerTimeout ' +
        'setTriggerType setType setUnconscious setUnitAbility setUnitLoadout setUnitPos setUnitPosWeak ' +
        'setUnitRank setUnitRecoilCoefficient setUnitTrait setUnloadInCombat setUserActionText setVariable ' +
        'setVectorDir setVectorDirAndUp setVectorUp setVehicleAmmo setVehicleAmmoDef setVehicleArmor ' +
        'setVehicleCargo setVehicleId setVehicleLock setVehiclePosition setVehicleTiPars setVehicleVarName ' +
        'setVelocity setVelocityTransformation setViewDistance setVisibleIfTreeCollapsed setWaves ' +
        'setWaypointBehaviour setWaypointCombatMode setWaypointCompletionRadius setWaypointDescription ' +
        'setWaypointForceBehaviour setWaypointFormation setWaypointHousePosition setWaypointLoiterRadius ' +
        'setWaypointLoiterType setWaypointName setWaypointPosition setWaypointScript setWaypointSpeed ' +
        'setWaypointStatements setWaypointTimeout setWaypointType setWaypointVisible ' +
        'setWeaponReloadingTime setWind setWindDir setWindForce setWindStr setWPPos show3DIcons showChat ' +
        'showCinemaBorder showCommandingMenu showCompass showCuratorCompass showGPS showHUD showLegend ' +
        'showMap shownArtilleryComputer shownChat shownCompass shownCuratorCompass showNewEditorObject ' +
        'shownGPS shownHUD shownMap shownPad shownRadio shownScoretable shownUAVFeed shownWarrant ' +
        'shownWatch showPad showRadio showScoretable showSubtitles showUAVFeed showWarrant showWatch ' +
        'showWaypoint showWaypoints side sideAmbientLife sideChat sideEmpty sideEnemy sideFriendly ' +
        'sideLogic sideRadio sideUnknown simpleTasks simulationEnabled simulCloudDensity ' +
        'simulCloudOcclusion simulInClouds simulWeatherSync sin size sizeOf skill skillFinal skipTime ' +
        'sleep sliderPosition sliderRange sliderSetPosition sliderSetRange sliderSetSpeed sliderSpeed ' +
        'slingLoadAssistantShown soldierMagazines someAmmo sort soundVolume spawn speaker speed speedMode ' +
        'splitString sqrt squadParams stance startLoadingScreen step stop stopEngineRTD stopped str ' +
        'sunOrMoon supportInfo suppressFor surfaceIsWater surfaceNormal surfaceType swimInDepth ' +
        'switchableUnits switchAction switchCamera switchGesture switchLight switchMove ' +
        'synchronizedObjects synchronizedTriggers synchronizedWaypoints synchronizeObjectsAdd ' +
        'synchronizeObjectsRemove synchronizeTrigger synchronizeWaypoint systemChat systemOfUnits tan ' +
        'targetKnowledge targetsAggregate targetsQuery taskAlwaysVisible taskChildren taskCompleted ' +
        'taskCustomData taskDescription taskDestination taskHint taskMarkerOffset taskNull taskParent ' +
        'taskResult taskState taskType teamMember teamMemberNull teamName teams teamSwitch ' +
        'teamSwitchEnabled teamType terminate terrainIntersect terrainIntersectASL text textLog ' +
        'textLogFormat tg time timeMultiplier titleCut titleFadeOut titleObj titleRsc titleText toArray ' +
        'toFixed toLower toString toUpper triggerActivated triggerActivation triggerArea ' +
        'triggerAttachedVehicle triggerAttachObject triggerAttachVehicle triggerStatements triggerText ' +
        'triggerTimeout triggerTimeoutCurrent triggerType turretLocal turretOwner turretUnit tvAdd tvClear ' +
        'tvCollapse tvCount tvCurSel tvData tvDelete tvExpand tvPicture tvSetCurSel tvSetData tvSetPicture ' +
        'tvSetPictureColor tvSetPictureColorDisabled tvSetPictureColorSelected tvSetPictureRight ' +
        'tvSetPictureRightColor tvSetPictureRightColorDisabled tvSetPictureRightColorSelected tvSetText ' +
        'tvSetTooltip tvSetValue tvSort tvSortByValue tvText tvTooltip tvValue type typeName typeOf ' +
        'UAVControl uiNamespace uiSleep unassignCurator unassignItem unassignTeam unassignVehicle ' +
        'underwater uniform uniformContainer uniformItems uniformMagazines unitAddons unitAimPosition ' +
        'unitAimPositionVisual unitBackpack unitIsUAV unitPos unitReady unitRecoilCoefficient units ' +
        'unitsBelowHeight unlinkItem unlockAchievement unregisterTask updateDrawIcon updateMenuItem ' +
        'updateObjectTree useAISteeringComponent useAudioTimeForMoves vectorAdd vectorCos ' +
        'vectorCrossProduct vectorDiff vectorDir vectorDirVisual vectorDistance vectorDistanceSqr ' +
        'vectorDotProduct vectorFromTo vectorMagnitude vectorMagnitudeSqr vectorMultiply vectorNormalized ' +
        'vectorUp vectorUpVisual vehicle vehicleCargoEnabled vehicleChat vehicleRadio vehicles ' +
        'vehicleVarName velocity velocityModelSpace verifySignature vest vestContainer vestItems ' +
        'vestMagazines viewDistance visibleCompass visibleGPS visibleMap visiblePosition ' +
        'visiblePositionASL visibleScoretable visibleWatch waves waypointAttachedObject ' +
        'waypointAttachedVehicle waypointAttachObject waypointAttachVehicle waypointBehaviour ' +
        'waypointCombatMode waypointCompletionRadius waypointDescription waypointForceBehaviour ' +
        'waypointFormation waypointHousePosition waypointLoiterRadius waypointLoiterType waypointName ' +
        'waypointPosition waypoints waypointScript waypointsEnabledUAV waypointShow waypointSpeed ' +
        'waypointStatements waypointTimeout waypointTimeoutCurrent waypointType waypointVisible ' +
        'weaponAccessories weaponAccessoriesCargo weaponCargo weaponDirection weaponInertia weaponLowered ' +
        'weapons weaponsItems weaponsItemsCargo weaponState weaponsTurret weightRTD west WFSideText wind',
      literal:
        'true false nil'
    },
    contains: [
      hljs.C_LINE_COMMENT_MODE,
      hljs.C_BLOCK_COMMENT_MODE,
      hljs.NUMBER_MODE,
      VARIABLE,
      FUNCTION,
      STRINGS,
      CPP.preprocessor
    ],
    illegal: /#/
  };
};