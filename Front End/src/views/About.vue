<template>
    <div class="about">
      <h1>About Team Mario</h1>
      
      <section class="about-section">
        <h2>Our Mission</h2>
        <p>
          Team Mario is dedicated to making machine learning accessible to everyone. 
          Our image classification system allows users to easily identify animals in images
          using state-of-the-art ML models, without requiring technical expertise.
        </p>
      </section>
      
      <section class="about-section">
        <h2>Our Team</h2>
        <div class="team-grid">
          <div class="team-member">
            <div class="member-avatar">
              <img src="@/assets/team/miguel.jpg" alt="Miguel Mario Mendez Alcaraz"/>
            </div>
            <h3>Miguel Mario Mendez Alcaraz</h3>
            <p class="member-role">Project Manager</p>
            <p>Leading project development</p>
          </div>
          
          <div class="team-member">
            <div class="member-avatar">
              <img src="@/assets/team/angel.jpg" alt="Ángel David Cortés Pacheco"/>
            </div>
            <h3>Ángel David Cortés Pacheco</h3>
            <p class="member-role">Backend/Middleware Developer</p>
            <p>In charge of designing the API and maintaining cloud infrastructure</p>
          </div>
          
          <div class="team-member">
            <div class="member-avatar">
              <img src="@/assets/team/dion.jpg" alt="Dion Rizo Velarde"/>
            </div>
            <h3>Dion Rizo Velarde</h3>
            <p class="member-role">MLOps</p>
            <p>Creating models and supervising their accuracy periodically</p>
          </div>

          <div class="team-member">
            <div class="member-avatar">
              <img src="@/assets/team/marco.jpg" alt="Marco Albanese"/>
            </div>
            <h3>Marco Albanese</h3>
            <p class="member-role">Frontend Developer</p>
            <p>Website designer</p>
          </div>

          <div class="team-member">
            <div class="member-avatar">
              <img src="@/assets/team/michelle.jpg" alt="Michelle Figueroa Jacques"/>
            </div>
            <h3>Michelle Figueroa Jacques</h3>
            <p class="member-role">Tester</p>
            <p>Unit testing, functional validation and GitActions maintainer</p>
          </div>
        </div>
      </section>
      
      <section class="about-section">
        <h2>Technology</h2>
        <p>
          Our image classification system is built on a deep learning model which 
          achieves >90% accuracy using various architectures (ResNet18, ResNet34, 
          MobileNet, EfficientNet) and has been optimized for quick training while 
          maintaining high accuracy. The frontend is built with Vue.js and the 
          backend utilizes Python with FastAPI.
        </p>
        
        <div class="tech-stack">
          <div class="tech-item">
            <h4>Frontend</h4>
            <ul>
              <li>Vue.js 3</li>
              <li>Vuex</li>
              <li>Vue Router</li>
              <li>Modern CSS</li>
            </ul>
          </div>
          
          <div class="tech-item">
            <h4>Backend</h4>
            <ul>
              <li>Python 3.9+</li>
              <li>FastAPI</li>
            </ul>
          </div>
          
          <div class="tech-item">
            <h4>ML Stack</h4>
            <ul>
              <li>PyTorch</li>
              <li>MLFlow</li>
              <li>Git & DVC</li>
            </ul>
          </div>
        </div>
      </section>
      
      <section class="about-section">
        <h2>Resources</h2>
        <div class="resource-links">
          <a href="https://github.com/" class="resource-link">
            <i class="resource-icon">📂</i>
            <span>Site GitHub Repository</span>
          </a>

          <a href="https://github.com/DionRizo/animal_classification_model" class="resource-link">
            <i class="resource-icon">📂</i>
            <span>ML Model Repository</span>
          </a>
        </div>
      </section>
      
      <section class="about-section">
        <h2>Contact Us</h2>
        <p>
          Have questions, suggestions, or need support? We'd love to hear from you!
        </p>
        <p class="contact-email">
          📧 <a href="mailto:marco.albanese@iteso.mx">marco.albanese@iteso.mx</a> <br>
          📧 <a href="is726442@iteso.mx">is726442@iteso.mx (Ángel Cortés)</a> <br>
          📧 <a href="mailto:is727328@iteso.mx">is727328@iteso.mx (Michelle Figueroa)</a> <br>
          📧 <a href="mailto:is727312@iteso.mx">is727312@iteso.mx (Miguel Mendez)</a> <br>
          📧 <a href="mailto:dion.rizo@iteso.mx">dion.rizo@iteso.mx</a>
        </p>
      </section>
    </div>
  </template>
  
  <style scoped>
  .about {
    max-width: 800px;
    margin: 0 auto;
  }
  
  .about-section {
    margin-bottom: 3rem;
  }
  
  .about-section h2 {
    border-bottom: 2px solid #eee;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
  }
  
  .team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 2rem;
  }
  
  .team-member {
    text-align: center;
  }
  
  .member-avatar {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background-color: #2c3e50;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    font-weight: bold;
    margin: 0 auto 1rem;
  }
  
  .member-avatar img {
    border-radius: 50%;
  }

  .member-role {
    color: #666;
    font-style: italic;
    margin-bottom: 0.5rem;
  }
  
  .tech-stack {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
  }
  
  .tech-item {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 1.5rem;
  }
  
  .tech-item h4 {
    margin-top: 0;
    margin-bottom: 1rem;
    color: #42b983;
  }
  
  .tech-item ul {
    padding-left: 1.5rem;
    margin: 0;
  }
  
  .tech-item li {
    margin-bottom: 0.5rem;
  }
  
  .resource-links {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 1rem;
  }
  
  .resource-link {
    display: flex;
    align-items: center;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 8px;
    text-decoration: none;
    color: #333;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  
  .resource-link:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
  
  .resource-icon {
    font-size: 1.5rem;
    margin-right: 0.75rem;
  }
  
  .contact-email {
    margin-top: 1.5rem;
    font-size: 1.1rem;
    font-size: 1.1rem;
}

.contact-email a {
  color: #42b983;
  text-decoration: none;
}

.contact-email a:hover {
  text-decoration: underline;
}

@media (max-width: 600px) {
  .team-grid, .tech-stack, .resource-links {
    grid-template-columns: 1fr;
  }
}
</style>
  