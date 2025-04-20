<template>
  <div class="app">
    <header>
      <nav>
        <div class="logo">
          <img src="./assets/logo.png" alt="ML Vision" />
          <h1>ML Vision</h1>
        </div>
        <div class="nav-links">
          <router-link to="/" v-if="isAuthenticated">Home</router-link>
          <router-link to="/history" v-if="isAuthenticated">History</router-link>
          <router-link to="/about">About</router-link>
          <a href="#" @click.prevent="logout" v-if="isAuthenticated">Logout</a>
          <router-link to="/login" v-else>Login</router-link>
        </div>
      </nav>
    </header>
    
    <main>
      <router-view></router-view>
    </main>
    
    <footer>
      <p>&copy; 2025 Team Mario. All rights reserved.</p>
    </footer>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useStore } from 'vuex'
import { useRouter } from 'vue-router'

export default {
  setup() {
    const store = useStore()
    const router = useRouter()
    
    const isAuthenticated = computed(() => store.state.isAuthenticated)
    
    const logout = () => {
      store.dispatch('logout')
      router.push('/login')
    }
    
    return {
      isAuthenticated,
      logout
    }
  }
}
</script>

<style>
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

header {
  background-color: #2c3e50;
  color: white;
  padding: 1rem;
}

nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.logo img {
  height: 2rem;
}

.nav-links {
  display: flex;
  gap: 1.5rem;
}

.nav-links a {
  color: white;
  text-decoration: none;
  font-weight: 500;
}

.nav-links a:hover {
  text-decoration: underline;
}

main {
  flex: 1;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

footer {
  background-color: #2c3e50;
  color: white;
  text-align: center;
  padding: 1rem;
  margin-top: auto;
}
</style>