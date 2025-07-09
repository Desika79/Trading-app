# 🚀 Quick Deploy - AI Trading System

## ❌ Netlify Issue Fixed
**Problem**: Netlify is for static sites, not Python ML applications.
**Solution**: Use proper Python hosting platforms below.

---

## ⚡ 1-Click Deploy Options

### 🚂 Railway (Recommended)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

1. Click button above
2. Connect GitHub repo
3. Set environment variables:
   ```
   BINANCE_API_KEY=your_key
   BINANCE_SECRET_KEY=your_secret
   SECRET_KEY=random-string-here
   ```
4. Deploy automatically!

**Cost**: $5-20/month

---

### 🎨 Render (Free Tier)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

1. Sign up at render.com
2. Connect GitHub repo
3. Render auto-detects `render.yaml`
4. Add environment variables in dashboard
5. Deploy!

**Cost**: Free tier available

---

### 🟪 Heroku (Traditional)
```bash
# Install Heroku CLI then:
heroku create your-app-name
heroku addons:create heroku-postgresql:mini
heroku addons:create heroku-redis:mini
git push heroku main
```

**Cost**: $25+/month

---

## 🔧 Required Environment Variables

**Minimum Setup:**
```env
SECRET_KEY=your-random-secret-key-change-this
MIN_WIN_RATE=0.80
LOG_LEVEL=INFO
```

**For Live Trading Data:**
```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
DERIV_API_TOKEN=your_deriv_token
TWELVE_DATA_API_KEY=your_twelvedata_key
```

**Database URLs** (auto-provided by platforms):
```env
POSTGRES_URL=postgresql://...
REDIS_URL=redis://...
```

---

## 🎯 After Deployment

1. **Test Health Check**:
   ```bash
   curl https://your-app-url.com/health
   ```

2. **Generate First Signal**:
   ```bash
   curl -X POST "https://your-app-url.com/signals/generate/advanced?symbol=EURUSD"
   ```

3. **Access Dashboard**:
   - Visit: `https://your-app-url.com`
   - WebSocket: `wss://your-app-url.com/ws`
   - API Docs: `https://your-app-url.com/docs`

---

## 🆘 Common Issues

**"Failed to build"**
- Check Python version (should be 3.11)
- Verify requirements.txt exists

**"Application error"**
- Check environment variables
- View deployment logs

**"No signals generated"**
- Add API keys for data providers
- Check minimum win rate settings

---

## 📞 Support

- **Issues**: Check DEPLOYMENT_GUIDE.md
- **Logs**: Use platform dashboard
- **Health**: Visit `/health` endpoint

**🎉 Your AI Trading System will be live in 5 minutes!**