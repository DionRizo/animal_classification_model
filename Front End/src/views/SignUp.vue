<template>
    <div class="signup-container">
      <div class="signup-card">
        <div class="signup-header">
          <h1>Create Account</h1>
          <p>Sign up to start using ML Vision</p>
        </div>
  
        <form @submit.prevent="handleSignup" class="signup-form">
          <div class="form-group">
            <label for="username">Username</label>
            <input
              id="username"
              type="text"
              v-model="form.username"
              required
              placeholder="Enter your username"
            />
          </div>
  
          <div class="form-group">
            <label for="email">Email</label>
            <input
              id="email"
              type="email"
              v-model="form.email"
              required
              placeholder="Enter your email"
            />
          </div>
  
          <div class="form-group">
            <label for="password">Password</label>
            <input
              id="password"
              type="password"
              v-model="form.password"
              required
              placeholder="Enter your password"
            />
          </div>
  
          <div class="form-group">
            <label for="confirmPassword">Confirm Password</label>
            <input
              id="confirmPassword"
              type="password"
              v-model="form.confirmPassword"
              required
              placeholder="Re‑enter your password"
            />
          </div>
  
          <div class="form-actions">
            <button type="submit" class="signup-btn" :disabled="isLoading">
              {{ isLoading ? 'Signing up...' : 'Sign Up' }}
            </button>
          </div>
  
          <div class="error-message" v-if="error">
            {{ error }}
          </div>
          <div class="success-message" v-if="success">
            {{ success }}
          </div>
        </form>
  
        <div class="signup-footer">
          <p>
            Already have an account?
            <router-link to="/login">Login</router-link>
          </p>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import { reactive, ref } from 'vue'
  import { useRouter } from 'vue-router'
  
  export default {
    setup() {
      const router = useRouter()
      const form = reactive({
        username: '',
        email: '',
        password: '',
        confirmPassword: ''
      })
      const isLoading = ref(false)
      const error = ref(null)
      const success = ref(null)
  
      const handleSignup = async () => {
        // simple validation
        if (!form.username || !form.email || !form.password || !form.confirmPassword) {
          error.value = 'Please fill in all fields'
          return
        }
        if (form.password !== form.confirmPassword) {
          error.value = 'Passwords do not match'
          return
        }
  
        error.value = null
        isLoading.value = true
  
        try {
          // In a real app, call your signup API here, e.g.
          // await store.dispatch('signup', form)
  
          // simulate network delay
          await new Promise((resolve) => setTimeout(resolve, 1000))
  
          success.value = 'Account created! Redirecting to login…'
          setTimeout(() => {
            router.push('/login')
          }, 1500)
        } catch (e) {
          error.value = 'Failed to create account. Please try again.'
        } finally {
          isLoading.value = false
        }
      }
  
      return {
        form,
        isLoading,
        error,
        success,
        handleSignup
      }
    }
  }
  </script>
  
  <style scoped>
  .signup-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
  }
  
  .signup-card {
    background: white;
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    width: 100%;
    max-width: 400px;
    overflow: hidden;
  }
  
  .signup-header {
    text-align: center;
    padding: 2rem 1.5rem;
  }
  
  .signup-header h1 {
    margin-bottom: 0.5rem;
    color: #2c3e50;
  }
  
  .signup-header p {
    color: #666;
  }
  
  .signup-form {
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
  
  .signup-btn {
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
  
  .signup-btn:hover:not(:disabled) {
    background-color: #3aa876;
  }
  
  .signup-btn:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
  
  .error-message {
    color: #e74c3c;
    margin-top: 1rem;
    text-align: center;
    font-size: 0.9rem;
  }
  
  .success-message {
    color: #27ae60;
    margin-top: 1rem;
    text-align: center;
    font-size: 0.9rem;
  }
  
  .signup-footer {
    text-align: center;
    padding: 1.5rem;
    background-color: #f8f9fa;
    border-top: 1px solid #eee;
  }
  
  .signup-footer a {
    color: #42b983;
    text-decoration: none;
    font-weight: 500;
  }
  
  .signup-footer a:hover {
    text-decoration: underline;
  }
  </style>
  