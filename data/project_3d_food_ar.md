# Project: 3D Food Visualization App

I was the **team lead** on this project. It is an augmented reality (AR) restaurant platform built with React Native and Expo Router on the frontend, and FastAPI with PostgreSQL on the backend.

## The idea

Menus with tiny photos don't tell you much about what will actually arrive at your table. In this app, a customer scans a QR code at the restaurant and can see a life-size 3D model of a dish sitting right on their table through their phone camera, before ordering it. The 3D dishes are stored in .glb and .usdz formats, which are common 3D model formats for AR on mobile devices.

## Four portals with role-based access

The backend is written in Python with FastAPI and stores data in PostgreSQL. It uses role-based access control (RBAC), which means each type of user only sees and does what their role allows. There are four stakeholder portals:

- **Customer:** browse the menu, view dishes in AR, add to cart, order and pay.
- **Content Admin:** manage the menu and the 3D dish content.
- **Order Manager:** handle incoming orders.
- **Super Admin:** oversee the whole platform.

## Smart inventory

The app tracks inventory in real time. When an ingredient runs out, menu items that depend on it are automatically hidden or disabled, so customers can't order a dish the kitchen can't make.

## Ordering, payments and analytics

I designed the end-to-end ordering flow: menu, cart, checkout and payment. The platform also has built-in analytics that track revenue, ratings, and how many times each dish was viewed in AR, so a restaurant can see which dishes people look at and which ones they actually buy.

## In plain words

This is a phone app for restaurants that lets diners see a realistic, full-size model of their meal on the table before they order. Behind it is a system with separate screens for customers, menu editors, order staff and the owner, which also keeps track of stock and sales.
