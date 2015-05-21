// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// Notification identifies a GitHub notification for a user.
type Notification struct {
	ID         *string              `json:"id,omitempty"`
	Repository *Repository          `json:"repository,omitempty"`
	Subject    *NotificationSubject `json:"subject,omitempty"`

	// Reason identifies the event that triggered the notification.
	//
	// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#notification-reasons
	Reason *string `json:"reason,omitempty"`

	Unread     *bool      `json:"unread,omitempty"`
	UpdatedAt  *time.Time `json:"updated_at,omitempty"`
	LastReadAt *time.Time `json:"last_read_at,omitempty"`
	URL        *string    `json:"url,omitempty"`
}

// NotificationSubject identifies the subject of a notification.
type NotificationSubject struct {
	Title            *string `json:"title,omitempty"`
	URL              *string `json:"url,omitempty"`
	LatestCommentURL *string `json:"latest_comment_url,omitempty"`
	Type             *string `json:"type,omitempty"`
}

// NotificationListOptions specifies the optional parameters to the
// ActivityService.ListNotifications method.
type NotificationListOptions struct {
	All           bool      `url:"all,omitempty"`
	Participating bool      `url:"participating,omitempty"`
	Since         time.Time `url:"since,omitempty"`
}

// ListNotifications lists all notifications for the authenticated user.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#list-your-notifications
func (s *ActivityService) ListNotifications(opt *NotificationListOptions) ([]Notification, *Response, error) {
	u := fmt.Sprintf("notifications")
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var notifications []Notification
	resp, err := s.client.Do(req, &notifications)
	if err != nil {
		return nil, resp, err
	}

	return notifications, resp, err
}

// ListRepositoryNotifications lists all notifications in a given repository
// for the authenticated user.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#list-your-notifications-in-a-repository
func (s *ActivityService) ListRepositoryNotifications(owner, repo string, opt *NotificationListOptions) ([]Notification, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/notifications", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var notifications []Notification
	resp, err := s.client.Do(req, &notifications)
	if err != nil {
		return nil, resp, err
	}

	return notifications, resp, err
}

type markReadOptions struct {
	LastReadAt time.Time `url:"last_read_at,omitempty"`
}

// MarkNotificationsRead marks all notifications up to lastRead as read.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#mark-as-read
func (s *ActivityService) MarkNotificationsRead(lastRead time.Time) (*Response, error) {
	u := fmt.Sprintf("notifications")
	u, err := addOptions(u, markReadOptions{lastRead})
	if err != nil {
		return nil, err
	}

	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// MarkRepositoryNotificationsRead marks all notifications up to lastRead in
// the specified repository as read.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#mark-notifications-as-read-in-a-repository
func (s *ActivityService) MarkRepositoryNotificationsRead(owner, repo string, lastRead time.Time) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/notifications", owner, repo)
	u, err := addOptions(u, markReadOptions{lastRead})
	if err != nil {
		return nil, err
	}

	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// GetThread gets the specified notification thread.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#view-a-single-thread
func (s *ActivityService) GetThread(id string) (*Notification, *Response, error) {
	u := fmt.Sprintf("notifications/threads/%v", id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	notification := new(Notification)
	resp, err := s.client.Do(req, notification)
	if err != nil {
		return nil, resp, err
	}

	return notification, resp, err
}

// MarkThreadRead marks the specified thread as read.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#mark-a-thread-as-read
func (s *ActivityService) MarkThreadRead(id string) (*Response, error) {
	u := fmt.Sprintf("notifications/threads/%v", id)

	req, err := s.client.NewRequest("PATCH", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// GetThreadSubscription checks to see if the authenticated user is subscribed
// to a thread.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#get-a-thread-subscription
func (s *ActivityService) GetThreadSubscription(id string) (*Subscription, *Response, error) {
	u := fmt.Sprintf("notifications/threads/%v/subscription", id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	sub := new(Subscription)
	resp, err := s.client.Do(req, sub)
	if err != nil {
		return nil, resp, err
	}

	return sub, resp, err
}

// SetThreadSubscription sets the subscription for the specified thread for the
// authenticated user.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#set-a-thread-subscription
func (s *ActivityService) SetThreadSubscription(id string, subscription *Subscription) (*Subscription, *Response, error) {
	u := fmt.Sprintf("notifications/threads/%v/subscription", id)

	req, err := s.client.NewRequest("PUT", u, subscription)
	if err != nil {
		return nil, nil, err
	}

	sub := new(Subscription)
	resp, err := s.client.Do(req, sub)
	if err != nil {
		return nil, resp, err
	}

	return sub, resp, err
}

// DeleteThreadSubscription deletes the subscription for the specified thread
// for the authenticated user.
//
// GitHub API Docs: https://developer.github.com/v3/activity/notifications/#delete-a-thread-subscription
func (s *ActivityService) DeleteThreadSubscription(id string) (*Response, error) {
	u := fmt.Sprintf("notifications/threads/%v/subscription", id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
