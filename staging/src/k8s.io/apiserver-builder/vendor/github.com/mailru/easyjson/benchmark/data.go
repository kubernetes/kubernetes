// Package benchmark provides a simple benchmark for easyjson against default serialization and ffjson.
// The data example is taken from https://dev.twitter.com/rest/reference/get/search/tweets
package benchmark

import (
	"io/ioutil"
)

var largeStructText, _ = ioutil.ReadFile("example.json")
var xlStructData XLStruct

func init() {
	for i := 0; i < 50; i++ {
		xlStructData.Data = append(xlStructData.Data, largeStructData)
	}
}

var smallStructText = []byte(`{"hashtags":[{"indices":[5, 10],"text":"some-text"}],"urls":[],"user_mentions":[]}`)
var smallStructData = Entities{
	Hashtags:     []Hashtag{{Indices: []int{5, 10}, Text: "some-text"}},
	Urls:         []*string{},
	UserMentions: []*string{},
}

type SearchMetadata struct {
	CompletedIn float64 `json:"completed_in"`
	Count       int     `json:"count"`
	MaxID       int     `json:"max_id"`
	MaxIDStr    string  `json:"max_id_str"`
	NextResults string  `json:"next_results"`
	Query       string  `json:"query"`
	RefreshURL  string  `json:"refresh_url"`
	SinceID     int     `json:"since_id"`
	SinceIDStr  string  `json:"since_id_str"`
}

type Hashtag struct {
	Indices []int  `json:"indices"`
	Text    string `json:"text"`
}

//easyjson:json
type Entities struct {
	Hashtags     []Hashtag `json:"hashtags"`
	Urls         []*string `json:"urls"`
	UserMentions []*string `json:"user_mentions"`
}

type UserEntityDescription struct {
	Urls []*string `json:"urls"`
}

type URL struct {
	ExpandedURL *string `json:"expanded_url"`
	Indices     []int   `json:"indices"`
	URL         string  `json:"url"`
}

type UserEntityURL struct {
	Urls []URL `json:"urls"`
}

type UserEntities struct {
	Description UserEntityDescription `json:"description"`
	URL         UserEntityURL         `json:"url"`
}

type User struct {
	ContributorsEnabled            bool         `json:"contributors_enabled"`
	CreatedAt                      string       `json:"created_at"`
	DefaultProfile                 bool         `json:"default_profile"`
	DefaultProfileImage            bool         `json:"default_profile_image"`
	Description                    string       `json:"description"`
	Entities                       UserEntities `json:"entities"`
	FavouritesCount                int          `json:"favourites_count"`
	FollowRequestSent              *string      `json:"follow_request_sent"`
	FollowersCount                 int          `json:"followers_count"`
	Following                      *string      `json:"following"`
	FriendsCount                   int          `json:"friends_count"`
	GeoEnabled                     bool         `json:"geo_enabled"`
	ID                             int          `json:"id"`
	IDStr                          string       `json:"id_str"`
	IsTranslator                   bool         `json:"is_translator"`
	Lang                           string       `json:"lang"`
	ListedCount                    int          `json:"listed_count"`
	Location                       string       `json:"location"`
	Name                           string       `json:"name"`
	Notifications                  *string      `json:"notifications"`
	ProfileBackgroundColor         string       `json:"profile_background_color"`
	ProfileBackgroundImageURL      string       `json:"profile_background_image_url"`
	ProfileBackgroundImageURLHTTPS string       `json:"profile_background_image_url_https"`
	ProfileBackgroundTile          bool         `json:"profile_background_tile"`
	ProfileImageURL                string       `json:"profile_image_url"`
	ProfileImageURLHTTPS           string       `json:"profile_image_url_https"`
	ProfileLinkColor               string       `json:"profile_link_color"`
	ProfileSidebarBorderColor      string       `json:"profile_sidebar_border_color"`
	ProfileSidebarFillColor        string       `json:"profile_sidebar_fill_color"`
	ProfileTextColor               string       `json:"profile_text_color"`
	ProfileUseBackgroundImage      bool         `json:"profile_use_background_image"`
	Protected                      bool         `json:"protected"`
	ScreenName                     string       `json:"screen_name"`
	ShowAllInlineMedia             bool         `json:"show_all_inline_media"`
	StatusesCount                  int          `json:"statuses_count"`
	TimeZone                       string       `json:"time_zone"`
	URL                            *string      `json:"url"`
	UtcOffset                      int          `json:"utc_offset"`
	Verified                       bool         `json:"verified"`
}

type StatusMetadata struct {
	IsoLanguageCode string `json:"iso_language_code"`
	ResultType      string `json:"result_type"`
}

type Status struct {
	Contributors         *string        `json:"contributors"`
	Coordinates          *string        `json:"coordinates"`
	CreatedAt            string         `json:"created_at"`
	Entities             Entities       `json:"entities"`
	Favorited            bool           `json:"favorited"`
	Geo                  *string        `json:"geo"`
	ID                   int64          `json:"id"`
	IDStr                string         `json:"id_str"`
	InReplyToScreenName  *string        `json:"in_reply_to_screen_name"`
	InReplyToStatusID    *string        `json:"in_reply_to_status_id"`
	InReplyToStatusIDStr *string        `json:"in_reply_to_status_id_str"`
	InReplyToUserID      *string        `json:"in_reply_to_user_id"`
	InReplyToUserIDStr   *string        `json:"in_reply_to_user_id_str"`
	Metadata             StatusMetadata `json:"metadata"`
	Place                *string        `json:"place"`
	RetweetCount         int            `json:"retweet_count"`
	Retweeted            bool           `json:"retweeted"`
	Source               string         `json:"source"`
	Text                 string         `json:"text"`
	Truncated            bool           `json:"truncated"`
	User                 User           `json:"user"`
}

//easyjson:json
type LargeStruct struct {
	SearchMetadata SearchMetadata `json:"search_metadata"`
	Statuses       []Status       `json:"statuses"`
}

//easyjson:json
type XLStruct struct {
	Data []LargeStruct
}
