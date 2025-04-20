<template>
    <div class="login-container">
      <div class="login-card">
        <div class="login-header">
          <h1>Welcome Back</h1>
          <p>Sign in to continue to ML Vision</p>
        </div>
        
        <form @submit.prevent="handleLogin" class="login-form">
          <div class="form-group">
            <label for="username">Username</label>
            <input 
              type="text" 
              id="username" 
              v-model="credentials.username"
              required
              placeholder="Enter your username"
            />
          </div>
          
          <div class="form-group">
            <label for="password">Password</label>
            <input 
              type="password" 
              id="password" 
              v-model="credentials.password"
              required
              placeholder="Enter your password"
            />
          </div>
          
          <div class="form-actions">
            <button 
              type="submit" 
              class="login-btn"
              :disabled="isLoading"
            >
              {{ isLoading ? 'Signing in...' : 'Sign In' }}
            </button>
          </div>
          
          <div class="error-message" v-if="error">
            {{ error }}
          </div>
        </form>
        
        <div class="login-footer">
          <p>Don't have an account? <a href="#">Sign up</a></p>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import { ref, reactive } from 'vue'
  import { useStore } from 'vuex'
  import { useRouter } from 'vue-router'
  
  export default {
    setup() {
      const store = useStore()
      const router = useRouter()
      
      const credentials = reactive({
        username: '',
        password: ''
      })
      
      const isLoading = ref(false)
      const error = ref(null)
      
      const handleLogin = async () => {
        if (!credentials.username || !credentials.password) {
          error.value = 'Please enter both username and password'
          return
        }
        
        error.value = null
        isLoading.value = true
        
        try {
          // For demo, we'll accept any username/password
          // In a real app, this would validate against your backend
          await store.dispatch('login', credentials)
          router.push('/')
        } catch (err) {
          error.value = 'Invalid username or password'
        } finally {
          isLoading.value = false
        }
      }
      
      return {
        credentials,
        isLoading,
        error,
        handleLogin
      }
    }
  }
  </script>
  
  <style scoped>
  .login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
  }
  
  .login-card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    width: 100%;
    max-width: 400px;
    overflow: hidden;
  }
  
  .login-header {
    text-align: center;
    padding: 2rem 1.5rem;
  }
  
  .login-header h1 {
    margin-bottom: 0.5rem;
    color: #2c3e50;
  }
  
  .login-header p {
    color: #666;
  }
  
  .login-form {
    padding: 0 1.5rem 1.5rem;
  }
  
  .form-group {
    margin-bottom: 1.5rem;
  }
  
  .form-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: #333;
  }
  
  .form-group input {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 1rem;
  }
  
  .form-actions {
    margin-top: 2rem;
  }
  
  .login-btn {
    width: 100%;
    padding: 0.75rem;
    background-color: #42b983;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s;
  }
  
  .login-btn:hover:not(:disabled) {
    background-color: #3aa876;
  }
  
  .login-btn:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
  
  .error-message {
    color: #e74c3c;
    margin-top: 1rem;
    text-align: center;
    font-size: 0.9rem;
  }
  
  .login-footer {
    text-align: center;
    padding: 1.5rem;
    background-color: #f8f9fa;
    border-top: 1px solid #eee;
  }
  
  .login-footer a {
    color: #42b983;
    text-decoration: none;
    font-weight: 500;
  }
  
  .login-footer a:hover {
    text-decoration: underline;
  }
  </style>