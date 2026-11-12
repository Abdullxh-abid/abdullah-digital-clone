# Project: Cross-Platform Game Hub

The Game Hub is an adaptive, cross-platform collection of games built with React Native, Expo and Expo Router. It started as a mobile-only application, and I turned it into a responsive interface that works on desktop, tablet and mobile.

## Accessibility first

A big part of this project was accessibility. I added:

- **High Contrast mode**
- **XL text**
- **Color-blind modes**
- **Full keyboard navigation**, using number keys 1 to 9 and 1 to 7 to move around without a mouse or touch

Together these reduced usability barriers across 9 app modules.

## Architecture

The hub integrates **chess.js** for the chess game, and I built custom reusable UI components called **AppShell**, **InfoCard** and **AppButton** so every screen looks and behaves consistently. Scores and progress are saved through a centralized **AsyncStorage** service layer, so score tracking and persistence work the same way across all games.

## Better interaction design

Across 7 playable games, I redesigned how the app talks to the player. Disruptive popup alerts were replaced with inline validation and real-time status updates, and I added one-click state recovery with Undo and Restart buttons. Popups interrupt the flow of a game, so it's better to show feedback right where the player is looking.

## In plain words

This is a set of small games that works on phones, tablets and computers. I made it easier to use for people who have trouble seeing colours or small text, or who prefer a keyboard, and I replaced annoying pop-up messages with gentler hints and easy undo buttons.
