// Package gamesmanagement provides access to the Google Play Game Services Management API.
//
// See https://developers.google.com/games/services
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/gamesmanagement/v1management"
//   ...
//   gamesmanagementService, err := gamesmanagement.New(oauthHttpClient)
package gamesmanagement

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Always reference these packages, just in case the auto-generated code
// below doesn't.
var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace

const apiId = "gamesManagement:v1management"
const apiName = "gamesManagement"
const apiVersion = "v1management"
const basePath = "https://www.googleapis.com/games/v1management/"

// OAuth2 scopes used by this API.
const (
	// Share your Google+ profile information and view and manage your game
	// activity
	GamesScope = "https://www.googleapis.com/auth/games"

	// Know your basic profile info and list of people in your circles.
	PlusLoginScope = "https://www.googleapis.com/auth/plus.login"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Achievements = NewAchievementsService(s)
	s.Applications = NewApplicationsService(s)
	s.Players = NewPlayersService(s)
	s.Rooms = NewRoomsService(s)
	s.Scores = NewScoresService(s)
	s.TurnBasedMatches = NewTurnBasedMatchesService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Achievements *AchievementsService

	Applications *ApplicationsService

	Players *PlayersService

	Rooms *RoomsService

	Scores *ScoresService

	TurnBasedMatches *TurnBasedMatchesService
}

func NewAchievementsService(s *Service) *AchievementsService {
	rs := &AchievementsService{s: s}
	return rs
}

type AchievementsService struct {
	s *Service
}

func NewApplicationsService(s *Service) *ApplicationsService {
	rs := &ApplicationsService{s: s}
	return rs
}

type ApplicationsService struct {
	s *Service
}

func NewPlayersService(s *Service) *PlayersService {
	rs := &PlayersService{s: s}
	return rs
}

type PlayersService struct {
	s *Service
}

func NewRoomsService(s *Service) *RoomsService {
	rs := &RoomsService{s: s}
	return rs
}

type RoomsService struct {
	s *Service
}

func NewScoresService(s *Service) *ScoresService {
	rs := &ScoresService{s: s}
	return rs
}

type ScoresService struct {
	s *Service
}

func NewTurnBasedMatchesService(s *Service) *TurnBasedMatchesService {
	rs := &TurnBasedMatchesService{s: s}
	return rs
}

type TurnBasedMatchesService struct {
	s *Service
}

type AchievementResetAllResponse struct {
	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string gamesManagement#achievementResetAllResponse.
	Kind string `json:"kind,omitempty"`

	// Results: The achievement reset results.
	Results []*AchievementResetResponse `json:"results,omitempty"`
}

type AchievementResetResponse struct {
	// CurrentState: The current state of the achievement. This is the same
	// as the initial state of the achievement.
	// Possible values are:
	// -
	// "HIDDEN"- Achievement is hidden.
	// - "REVEALED" - Achievement is
	// revealed.
	// - "UNLOCKED" - Achievement is unlocked.
	CurrentState string `json:"currentState,omitempty"`

	// DefinitionId: The ID of an achievement for which player state has
	// been updated.
	DefinitionId string `json:"definitionId,omitempty"`

	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string gamesManagement#achievementResetResponse.
	Kind string `json:"kind,omitempty"`

	// UpdateOccurred: Flag to indicate if the requested update actually
	// occurred.
	UpdateOccurred bool `json:"updateOccurred,omitempty"`
}

type GamesPlayedResource struct {
	// AutoMatched: True if the player was auto-matched with the currently
	// authenticated user.
	AutoMatched bool `json:"autoMatched,omitempty"`

	// TimeMillis: The last time the player played the game in milliseconds
	// since the epoch in UTC.
	TimeMillis int64 `json:"timeMillis,omitempty,string"`
}

type HiddenPlayer struct {
	// HiddenTimeMillis: The time this player was hidden.
	HiddenTimeMillis int64 `json:"hiddenTimeMillis,omitempty,string"`

	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string gamesManagement#hiddenPlayer.
	Kind string `json:"kind,omitempty"`

	// Player: The player information.
	Player *Player `json:"player,omitempty"`
}

type HiddenPlayerList struct {
	// Items: The players.
	Items []*HiddenPlayer `json:"items,omitempty"`

	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string gamesManagement#hiddenPlayerList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The pagination token for the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type Player struct {
	// AvatarImageUrl: The base URL for the image that represents the
	// player.
	AvatarImageUrl string `json:"avatarImageUrl,omitempty"`

	// DisplayName: The name to display for the player.
	DisplayName string `json:"displayName,omitempty"`

	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string gamesManagement#player.
	Kind string `json:"kind,omitempty"`

	// LastPlayedWith: Details about the last time this player played a
	// multiplayer game with the currently authenticated player. Populated
	// for PLAYED_WITH player collection members.
	LastPlayedWith *GamesPlayedResource `json:"lastPlayedWith,omitempty"`

	// Name: An object representation of the individual components of the
	// player's name.
	Name *PlayerName `json:"name,omitempty"`

	// PlayerId: The ID of the player.
	PlayerId string `json:"playerId,omitempty"`
}

type PlayerName struct {
	// FamilyName: The family name (last name) of this player.
	FamilyName string `json:"familyName,omitempty"`

	// GivenName: The given name (first name) of this player.
	GivenName string `json:"givenName,omitempty"`
}

type PlayerScoreResetResponse struct {
	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string gamesManagement#playerScoreResetResponse.
	Kind string `json:"kind,omitempty"`

	// ResetScoreTimeSpans: The time spans of the updated score.
	// Possible
	// values are:
	// - "ALL_TIME" - The score is an all-time score.
	// -
	// "WEEKLY" - The score is a weekly score.
	// - "DAILY" - The score is a
	// daily score.
	ResetScoreTimeSpans []string `json:"resetScoreTimeSpans,omitempty"`
}

// method id "gamesManagement.achievements.reset":

type AchievementsResetCall struct {
	s             *Service
	achievementId string
	opt_          map[string]interface{}
}

// Reset: Resets the achievement with the given ID for the currently
// authenticated player. This method is only accessible to whitelisted
// tester accounts for your application.
func (r *AchievementsService) Reset(achievementId string) *AchievementsResetCall {
	c := &AchievementsResetCall{s: r.s, opt_: make(map[string]interface{})}
	c.achievementId = achievementId
	return c
}

func (c *AchievementsResetCall) Do() (*AchievementResetResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "achievements/{achievementId}/reset")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{achievementId}", url.QueryEscape(c.achievementId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AchievementResetResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Resets the achievement with the given ID for the currently authenticated player. This method is only accessible to whitelisted tester accounts for your application.",
	//   "httpMethod": "POST",
	//   "id": "gamesManagement.achievements.reset",
	//   "parameterOrder": [
	//     "achievementId"
	//   ],
	//   "parameters": {
	//     "achievementId": {
	//       "description": "The ID of the achievement used by this method.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "achievements/{achievementId}/reset",
	//   "response": {
	//     "$ref": "AchievementResetResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.achievements.resetAll":

type AchievementsResetAllCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// ResetAll: Resets all achievements for the currently authenticated
// player for your application. This method is only accessible to
// whitelisted tester accounts for your application.
func (r *AchievementsService) ResetAll() *AchievementsResetAllCall {
	c := &AchievementsResetAllCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *AchievementsResetAllCall) Do() (*AchievementResetAllResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "achievements/reset")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AchievementResetAllResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Resets all achievements for the currently authenticated player for your application. This method is only accessible to whitelisted tester accounts for your application.",
	//   "httpMethod": "POST",
	//   "id": "gamesManagement.achievements.resetAll",
	//   "path": "achievements/reset",
	//   "response": {
	//     "$ref": "AchievementResetAllResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.achievements.resetForAllPlayers":

type AchievementsResetForAllPlayersCall struct {
	s             *Service
	achievementId string
	opt_          map[string]interface{}
}

// ResetForAllPlayers: Resets the achievement with the given ID for the
// all players. This method is only available to user accounts for your
// developer console. Only draft achievements can be reset.
func (r *AchievementsService) ResetForAllPlayers(achievementId string) *AchievementsResetForAllPlayersCall {
	c := &AchievementsResetForAllPlayersCall{s: r.s, opt_: make(map[string]interface{})}
	c.achievementId = achievementId
	return c
}

func (c *AchievementsResetForAllPlayersCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "achievements/{achievementId}/resetForAllPlayers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{achievementId}", url.QueryEscape(c.achievementId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Resets the achievement with the given ID for the all players. This method is only available to user accounts for your developer console. Only draft achievements can be reset.",
	//   "httpMethod": "POST",
	//   "id": "gamesManagement.achievements.resetForAllPlayers",
	//   "parameterOrder": [
	//     "achievementId"
	//   ],
	//   "parameters": {
	//     "achievementId": {
	//       "description": "The ID of the achievement used by this method.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "achievements/{achievementId}/resetForAllPlayers",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.applications.listHidden":

type ApplicationsListHiddenCall struct {
	s             *Service
	applicationId string
	opt_          map[string]interface{}
}

// ListHidden: Get the list of players hidden from the given
// application. This method is only available to user accounts for your
// developer console.
func (r *ApplicationsService) ListHidden(applicationId string) *ApplicationsListHiddenCall {
	c := &ApplicationsListHiddenCall{s: r.s, opt_: make(map[string]interface{})}
	c.applicationId = applicationId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of player resources to return in the response, used for
// paging. For any response, the actual number of player resources
// returned may be less than the specified maxResults.
func (c *ApplicationsListHiddenCall) MaxResults(maxResults int64) *ApplicationsListHiddenCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The token returned
// by the previous request.
func (c *ApplicationsListHiddenCall) PageToken(pageToken string) *ApplicationsListHiddenCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *ApplicationsListHiddenCall) Do() (*HiddenPlayerList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "applications/{applicationId}/players/hidden")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{applicationId}", url.QueryEscape(c.applicationId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(HiddenPlayerList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get the list of players hidden from the given application. This method is only available to user accounts for your developer console.",
	//   "httpMethod": "GET",
	//   "id": "gamesManagement.applications.listHidden",
	//   "parameterOrder": [
	//     "applicationId"
	//   ],
	//   "parameters": {
	//     "applicationId": {
	//       "description": "The application being requested.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of player resources to return in the response, used for paging. For any response, the actual number of player resources returned may be less than the specified maxResults.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "15",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token returned by the previous request.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "applications/{applicationId}/players/hidden",
	//   "response": {
	//     "$ref": "HiddenPlayerList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.players.hide":

type PlayersHideCall struct {
	s             *Service
	applicationId string
	playerId      string
	opt_          map[string]interface{}
}

// Hide: Hide the given player's leaderboard scores from the given
// application. This method is only available to user accounts for your
// developer console.
func (r *PlayersService) Hide(applicationId string, playerId string) *PlayersHideCall {
	c := &PlayersHideCall{s: r.s, opt_: make(map[string]interface{})}
	c.applicationId = applicationId
	c.playerId = playerId
	return c
}

func (c *PlayersHideCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "applications/{applicationId}/players/hidden/{playerId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{applicationId}", url.QueryEscape(c.applicationId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{playerId}", url.QueryEscape(c.playerId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Hide the given player's leaderboard scores from the given application. This method is only available to user accounts for your developer console.",
	//   "httpMethod": "POST",
	//   "id": "gamesManagement.players.hide",
	//   "parameterOrder": [
	//     "applicationId",
	//     "playerId"
	//   ],
	//   "parameters": {
	//     "applicationId": {
	//       "description": "The application being requested.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "playerId": {
	//       "description": "A player ID. A value of me may be used in place of the authenticated player's ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "applications/{applicationId}/players/hidden/{playerId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.players.unhide":

type PlayersUnhideCall struct {
	s             *Service
	applicationId string
	playerId      string
	opt_          map[string]interface{}
}

// Unhide: Unhide the given player's leaderboard scores from the given
// application. This method is only available to user accounts for your
// developer console.
func (r *PlayersService) Unhide(applicationId string, playerId string) *PlayersUnhideCall {
	c := &PlayersUnhideCall{s: r.s, opt_: make(map[string]interface{})}
	c.applicationId = applicationId
	c.playerId = playerId
	return c
}

func (c *PlayersUnhideCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "applications/{applicationId}/players/hidden/{playerId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{applicationId}", url.QueryEscape(c.applicationId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{playerId}", url.QueryEscape(c.playerId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Unhide the given player's leaderboard scores from the given application. This method is only available to user accounts for your developer console.",
	//   "httpMethod": "DELETE",
	//   "id": "gamesManagement.players.unhide",
	//   "parameterOrder": [
	//     "applicationId",
	//     "playerId"
	//   ],
	//   "parameters": {
	//     "applicationId": {
	//       "description": "The application being requested.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "playerId": {
	//       "description": "A player ID. A value of me may be used in place of the authenticated player's ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "applications/{applicationId}/players/hidden/{playerId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.rooms.reset":

type RoomsResetCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// Reset: Reset all rooms for the currently authenticated player for
// your application. This method is only accessible to whitelisted
// tester accounts for your application.
func (r *RoomsService) Reset() *RoomsResetCall {
	c := &RoomsResetCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *RoomsResetCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "rooms/reset")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Reset all rooms for the currently authenticated player for your application. This method is only accessible to whitelisted tester accounts for your application.",
	//   "httpMethod": "POST",
	//   "id": "gamesManagement.rooms.reset",
	//   "path": "rooms/reset",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.scores.reset":

type ScoresResetCall struct {
	s             *Service
	leaderboardId string
	opt_          map[string]interface{}
}

// Reset: Reset scores for the specified leaderboard for the currently
// authenticated player. This method is only accessible to whitelisted
// tester accounts for your application.
func (r *ScoresService) Reset(leaderboardId string) *ScoresResetCall {
	c := &ScoresResetCall{s: r.s, opt_: make(map[string]interface{})}
	c.leaderboardId = leaderboardId
	return c
}

func (c *ScoresResetCall) Do() (*PlayerScoreResetResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "leaderboards/{leaderboardId}/scores/reset")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{leaderboardId}", url.QueryEscape(c.leaderboardId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PlayerScoreResetResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Reset scores for the specified leaderboard for the currently authenticated player. This method is only accessible to whitelisted tester accounts for your application.",
	//   "httpMethod": "POST",
	//   "id": "gamesManagement.scores.reset",
	//   "parameterOrder": [
	//     "leaderboardId"
	//   ],
	//   "parameters": {
	//     "leaderboardId": {
	//       "description": "The ID of the leaderboard.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "leaderboards/{leaderboardId}/scores/reset",
	//   "response": {
	//     "$ref": "PlayerScoreResetResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.scores.resetForAllPlayers":

type ScoresResetForAllPlayersCall struct {
	s             *Service
	leaderboardId string
	opt_          map[string]interface{}
}

// ResetForAllPlayers: Reset scores for the specified leaderboard for
// all players. This method is only available to user accounts for your
// developer console. Only draft leaderboards can be reset.
func (r *ScoresService) ResetForAllPlayers(leaderboardId string) *ScoresResetForAllPlayersCall {
	c := &ScoresResetForAllPlayersCall{s: r.s, opt_: make(map[string]interface{})}
	c.leaderboardId = leaderboardId
	return c
}

func (c *ScoresResetForAllPlayersCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "leaderboards/{leaderboardId}/scores/resetForAllPlayers")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{leaderboardId}", url.QueryEscape(c.leaderboardId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Reset scores for the specified leaderboard for all players. This method is only available to user accounts for your developer console. Only draft leaderboards can be reset.",
	//   "httpMethod": "POST",
	//   "id": "gamesManagement.scores.resetForAllPlayers",
	//   "parameterOrder": [
	//     "leaderboardId"
	//   ],
	//   "parameters": {
	//     "leaderboardId": {
	//       "description": "The ID of the leaderboard.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "leaderboards/{leaderboardId}/scores/resetForAllPlayers",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "gamesManagement.turnBasedMatches.reset":

type TurnBasedMatchesResetCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// Reset: Reset all turn-based match data for a user. This method is
// only accessible to whitelisted tester accounts for your application.
func (r *TurnBasedMatchesService) Reset() *TurnBasedMatchesResetCall {
	c := &TurnBasedMatchesResetCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *TurnBasedMatchesResetCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "turnbasedmatches/reset")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Reset all turn-based match data for a user. This method is only accessible to whitelisted tester accounts for your application.",
	//   "httpMethod": "POST",
	//   "id": "gamesManagement.turnBasedMatches.reset",
	//   "path": "turnbasedmatches/reset",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/games",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}
