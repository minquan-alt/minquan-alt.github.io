# 🚀 Quick Setup Guide

## Bước 1: Cài đặt Dependencies

```bash
cd portfolio
npm install
```

## Bước 2: Chạy Development Server

```bash
npm run dev
```

Server sẽ chạy tại: http://localhost:3000

## Bước 3: Tùy chỉnh nội dung

### 3.1. Thông tin cá nhân

Mở `public/index.html` và cập nhật:

```html
<!-- Hero Section -->
<h1>Quang Minh</h1>  <!-- Đổi tên của bạn -->
<h2>AI Engineer | Computer Vision | NLP</h2>  <!-- Đổi title -->
<p>Building end-to-end AI systems...</p>  <!-- Đổi mô tả -->
```

### 3.2. Social Links

```html
<a href="https://github.com/your-username" target="_blank">GitHub</a>
<a href="https://linkedin.com/in/your-profile" target="_blank">LinkedIn</a>
<a href="mailto:your-email@email.com">Email</a>
```

### 3.3. Thêm/Sửa Projects

Mở `public/js/project-data.js` và chỉnh sửa các project:

```javascript
1: {
    title: "Tên project của bạn",
    subtitle: "Mô tả ngắn",
    tags: ["Tech1", "Tech2"],
    // ... các phần khác
}
```

### 3.4. Thêm ảnh projects

Đặt ảnh vào folder `public/images/`:
- `project1.jpg`
- `project2.jpg`
- `project3.jpg`

Kích thước khuyến nghị: **800x500px**

### 3.5. CV/Resume

Đặt file CV của bạn vào: `/resume/cv.pdf`

Hoặc đổi link trong `index.html`:
```html
<a href="/path/to/your/cv.pdf" download>Download CV</a>
```

### 3.6. Đổi màu theme

Mở `public/css/style.css` và chỉnh sửa:

```css
:root {
    --bg-primary: #0a0a0a;         /* Màu nền chính */
    --accent-primary: #3b82f6;     /* Màu accent (xanh dương) */
    --text-primary: #e0e0e0;       /* Màu chữ */
}
```

## Bước 4: Deploy

### Option 1: Vercel (Dễ nhất)

```bash
npm install -g vercel
vercel
```

### Option 2: Heroku

```bash
# Tạo Procfile
echo "web: node server.js" > Procfile

# Deploy
heroku create
git push heroku main
```

### Option 3: VPS/Server riêng

```bash
# Trên server
npm install
npm start

# Hoặc dùng PM2
npm install -g pm2
pm2 start server.js --name portfolio
pm2 save
pm2 startup
```

## 🎯 Checklist trước khi deploy

- [ ] Đã đổi tên, title, mô tả cá nhân
- [ ] Đã cập nhật social links (GitHub, LinkedIn, Email)
- [ ] Đã thêm/chỉnh sửa projects trong `project-data.js`
- [ ] Đã thêm ảnh projects vào `public/images/`
- [ ] Đã test tất cả links và buttons
- [ ] Đã thêm file CV vào folder resume
- [ ] Đã test responsive trên mobile
- [ ] Đã test trên nhiều browsers (Chrome, Firefox, Safari)

## 💡 Tips

1. **Ảnh projects**: Sử dụng ảnh chất lượng cao, có thể là:
   - Screenshot demo
   - Architecture diagram
   - Results visualization
   - Hoặc generated images từ MidJourney/DALL-E

2. **Nội dung projects**: Viết theo format:
   - Problem (tại sao làm?)
   - Data (dữ liệu gì?)
   - Architecture (cách giải quyết?)
   - Experiments (kết quả thế nào?)
   - Deployment (deploy như thế nào?)

3. **SEO**: Cập nhật meta tags trong `index.html`:
```html
<meta name="description" content="Mô tả về bạn">
<title>Your Name - AI Engineer Portfolio</title>
```

4. **Analytics**: Thêm Google Analytics vào trước `</body>`:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=YOUR-GA-ID"></script>
```

## ❓ Troubleshooting

**Lỗi: Port 3000 đã được sử dụng**
```bash
# Đổi port
PORT=8000 npm start
```

**Lỗi: Module not found**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Images không hiển thị**
- Check đường dẫn file ảnh
- Đảm bảo ảnh ở trong folder `public/images/`
- File name phải khớp với code (case-sensitive)

## 📞 Cần trợ giúp?

Mở issue trên GitHub hoặc liên hệ qua email!

---

**Good luck với portfolio của bạn! 🎉**
