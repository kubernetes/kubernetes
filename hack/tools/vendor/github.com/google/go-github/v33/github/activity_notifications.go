// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
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
	// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity#notification-reasons
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
	Before        time.Time `url:"before,omitempty"`

	ListOptions
}

// ListNotifications lists all notifications for the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-notifications-for-the-authenticated-user
func (s *ActivityService) ListNotifications(ctx context.Context, opts *NotificationListOptions) ([]*Notification, *Response, error) {
	u := fmt.Sprintf("notifications")
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var notifications []*Notification
	resp, err := s.client.Do(ctx, req, &notifications)
	if err != nil {
		return nil, resp, err
	}

	return notifications, resp, nil
}

// ListRepositoryNotifications lists all notifications in a given repository
// for the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-repository-notifications-for-the-authenticated-user
func (s *ActivityService) ListRepositoryNotifications(ctx context.Context, owner, repo string, opts *NotificationListOptions) ([]*Notification, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/notifications", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var notifications []*Notification
	resp, err := s.client.Do(ctx, req, &notifications)
	if err != nil {
		return nil, resp, err
	}

	return notifications, resp, nil
}

type markReadOptions struct {
	LastReadAt time.Time `json:"last_read_at,omitempty"`
}

// MarkNotificationsRead marks all notifications up to lastRead as read.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity#mark-as-read
func (s *ActivityService) MarkNotificationsRead(ctx context.Context, lastRead time.Time) (*Response, error) {
	opts := &markReadOptions{
		LastReadAt: lastRead,
	}
	req, err := s.client.NewRequest("PUT", "notifications", opts)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// MarkRepositoryNotificationsRead marks all notifications up to lastRead in
// the specified repository as read.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#mark-repository-notifications-as-read
func (s *ActivityService) MarkRepositoryNotificationsRead(ctx context.Context, owner, repo string, lastRead time.Time) (*Response, error) {
	opts := &markReadOptions{
		LastReadAt: lastRead,
	}
	u := fmt.Sprintf("repos/%v/%v/notifications", owner, repo)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// GetThread gets the specified notification thread.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#get-a-thread
func (s *ActivityService) GetThread(ctx context.Context, id string) (*Notification, *Response, error) {
	u := fmt.Sprintf("notifications/threads/%v", id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	notification := new(Notification)
	resp, err := s.client.Do(ctx, req, notification)
	if err != nil {
		return nil, resp, err
	}

	return notification, resp, nil
}

// MarkThreadRead marks the specified thread as read.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#mark-a-thread-as-read
func (s *ActivityService) MarkThreadRead(ctx context.Context, id string) (*Response, error) {
	u := fmt.Sprintf("notifications/threads/%v", id)

	req, err := s.client.NewRequest("PATCH", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// GetThreadSubscription checks to see if the authenticated user is subscribed
// to a thread.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#get-a-thread-subscription-for-the-authenticated-user
func (s *ActivityService) GetThreadSubscription(ctx context.Context, id string) (*Subscription, *Response, error) {
	u := fmt.Sprintf("notifications/threads/%v/subscription", id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	sub := new(Subscription)
	resp, err := s.client.Do(ctx, req, sub)
	if err != nil {
		return nil, resp, err
	}

	return sub, resp, nil
}

// SetThreadSubscription sets the subscription for the specified thread for the
// authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#set-a-thread-subscription
func (s *ActivityService) SetThreadSubscription(ctx context.Context, id string, subscription *Subscription) (*Subscription, *Response, error) {
	u := fmt.Sprintf("notifications/threads/%v/subscription", id)

	req, err := s.client.NewRequest("PUT", u, subscription)
	if err != nil {
		return nil, nil, err
	}

	sub := new(Subscription)
	resp, err := s.client.Do(ctx, req, sub)
	if err != nil {
		return nil, resp, err
	}

	return sub, resp, nil
}

// DeleteThreadSubscription deletes the subscription for the specified thread
// for the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#delete-a-thread-subscription
func (s *ActivityService) DeleteThreadSubscription(ctx context.Context, id string) (*Response, error) {
	u := fmt.Sprintf("notifications/threads/%v/subscription", id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
