#include "ObjectState.hpp"
#include <random>
#include <algorithm>
#include <utility>

ObjectState::ObjectState(float xPos, float yPos, float xVel, float yVel, const vector<bool>& visibilityHistory, int type, size_t id) :
   timeSinceVisible(1),
   type(type),
   id(id),
   posVel{xPos, yPos, xVel, yVel},
   visHistorySize(visibilityHistory.size()),
   visHistory(1)
{
   size_t visIdx = 0;
   bool visSeen = false;
   for (size_t bit = 0; bit < 7; ++bit) {
      visHistory <<= 1;
      if (visIdx < visibilityHistory.size()) {
	 if (!visibilityHistory[visIdx] and !visSeen) {
	    ++timeSinceVisible;
	 }
	 visHistory |= visibilityHistory[visIdx];
      }
   }
}

ObjectState::ObjectState(float xPos, float yPos, size_t visHistorySize, int type, size_t id) :
   timeSinceVisible(1),
   type(type),
   id(id),
   posVel{xPos, yPos, 0, 0},
   visHistorySize(visHistorySize)
{
   visHistory = 0xc0; // 11000000 (visible now, but not before now)
}

ObjectState::ObjectState(istream& in, size_t propertiesPerObject) :
   timeSinceVisible(1),
   visHistorySize(propertiesPerObject - 4)
{
  
   //Input each float as 4 bytes
   for (size_t i = 0; i < 4; ++i) {
      char* bytes = reinterpret_cast<char*>(&posVel[i]);
      for (size_t j = 0; j < sizeof(float); ++j) {
	 in.get(bytes[j]);
      }
   }
   
   bool visSeen = false;
   char c;
   in.get(c);
   visHistory = c;

   unsigned char mask = 0x40; // 01000000
   while (mask > 0) {
      if (!visSeen) {
	 if (c & mask) {
	    visSeen = true;
	 } else {
	    ++timeSinceVisible;
	 }
      }
      mask >>= 1;
   }

   char t;
   in.get(t);
   type = (unsigned char)t - 128;

   in >> id;
}

float ObjectState::getXPos() const
{
   return posVel[0];
}

float ObjectState::getYPos() const
{
   return posVel[1];
}

float ObjectState::getXVel() const
{
   return posVel[2];
}

float ObjectState::getYVel() const
{
   return posVel[3];
}

bool ObjectState::getVisibility(size_t idx) const
{
   return visHistory & (0x40 >> idx);
}

size_t ObjectState::getVisHistorySize() const
{
   return visHistorySize;
}

int ObjectState::getType() const
{
   return type;
}

tuple<float, float> ObjectState::getPos() const
{
   return tuple<float, float>{getXPos(), getYPos()};
}

float ObjectState::getProperty(size_t idx) const
{
   if (idx < 4) {
      return posVel[idx];
   } else {
      return getVisibility(idx - 4);
   }
}

size_t ObjectState::getNumProperties() const
{
   return visHistorySize + 4;
}

size_t ObjectState::getID() const
{
   return id;
}

size_t ObjectState::getTimeSinceVisible() const
{
   return timeSinceVisible;
}

void ObjectState::update(float newXPos, float newYPos, bool visible)
{
   visHistory &= 0x7f;            // 01111111: Zero out leading bit
   visHistory >>= 1;              // Shift over to make room
   visHistory |= 0x80;            // 10000000: Reinstate the leading bit
   visHistory |= (visible << 6);  // Set in first position of current byte

   if(visible)
   {
      posVel[2] = (newXPos - posVel[0])/timeSinceVisible; // xVel
      posVel[3] = (newYPos - posVel[1])/timeSinceVisible; // yVel
      posVel[0] = newXPos; // xPos
      posVel[1] = newYPos; // yPos
      timeSinceVisible = 1;
   }
   else
   {
      timeSinceVisible++;
   }
}

void ObjectState::save(ostream& out) const
{
   //Output each float as 4 bytes
   for (size_t i = 0; i < 4; ++i) {
      const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&posVel[i]);
      for (size_t j = 0; j < sizeof(float); ++j) {
	 out.put(bytes[j]);
      }
   }
   out.put(visHistory);
   out.put(char(128 + type));
   out << " " << id;
}

tuple<float, float, float> ObjectState::addError(const tuple<float, float, float> &err)
{
   // TODO: we need to double check the logic of this part

   /**
    * It takes in a tuple that represent the variance and perturb the
    * current object state a bit based on the variance (assuming normal
    * distribution)
    * @param tuple<float, float, float>   Representing the variance of xPos, yPos, and
    *                                     visibility.
    */

   default_random_engine generator(rand());

   normal_distribution<float> xDistri(0, get<0>(err));
   normal_distribution<float> yDistri(0, get<1>(err));
   normal_distribution<float> visDistri(0, get<1>(err));

   float dX = xDistri(generator);
   float dY = yDistri(generator);
   float dVis = visDistri(generator);  // also perturb the probability a bit

   posVel[0] += dX;
   posVel[1] += dY;
   // visibility += dVis;                 // add the visibility variance to the probability

   /*
   uniform_real_distribution<float> rd(0.0,1.0);
   bool vis = (rd(generator) <= visibility);
   visHistory[0] &= 0xbf;       // 10111111: Zero out first bit
   visHistory[0] |= (vis << 6); // Set first bit
   */
   confine();

   return make_tuple(dX, dY, dVis);
}

void ObjectState::confine()
{
   /**
    * Set the positions and velocities within reasonable range.
    * (Screen size)
    */

   float& xPos = posVel[0];
   float& yPos = posVel[1];
   float& xVel = posVel[2];
   float& yVel = posVel[3];

   if (xPos < 0 || yPos < 0 || xPos > 210.0f || yPos > 160.0f ||
      xVel < -210.0f || yVel < -160.0f || yVel > 160.0f || xVel > 210.0f)
   {
      // cout << "oops" << endl;
   }

   xPos = max(xPos, 0.0f);
   yPos = max(yPos, 0.0f);
   xVel = max(xVel, -210.0f);
   yVel = max(yVel, -160.0f);

   xPos = min(xPos, 210.0f);
   yPos = min(yPos, 160.0f);
   xVel = min(xVel, 210.0f);
   yVel = min(yVel, 160.0f);
}

ObjectState& ObjectState::operator+=(const ObjectState& rhs)
{
/*   if (numAdded == 0) {
      // get identity from the first object it adds
      id = rhs.getID();
      type = rhs.getType();
      for (size_t i = 0; i < 4; ++i) {
	 posVel[i] = rhs.posVel[i];
      }
      }*/
   if (id == rhs.getID())
   {
      posVel[0] += rhs.getXPos();
      posVel[1] += rhs.getYPos();
      posVel[2] += rhs.getXVel();
      posVel[3] += rhs.getYVel();
      //visibility += rhs.getVisibility(0);
      timeSinceVisible += rhs.getTimeSinceVisible();

//      numAdded++;
   }
   return *this;
}

ObjectState& ObjectState::operator-=(const ObjectState& rhs)
{
   if (id == rhs.getID())
   {
      posVel[0] -= rhs.getXPos();
      posVel[1] -= rhs.getYPos();
      posVel[2] -= rhs.getXVel();
      posVel[3] -= rhs.getYVel();
      //visibility -= rhs.getVisibility(0);
      timeSinceVisible -= rhs.getTimeSinceVisible();
   }
   return *this;
}

ObjectState ObjectState::square() const
{
   ObjectState squared(*this);

   for (size_t i = 0; i < 4; ++i)
   {
      squared.posVel[i] *= squared.posVel[i];
   }
   //squared.visibility *= squared.visibility;
   squared.timeSinceVisible *= squared.timeSinceVisible;

   return squared;
}

ObjectState ObjectState::average() const
{
   ObjectState avg(*this);
/*
   avg.posVel[0] /= numAdded;
   avg.posVel[1] /= numAdded;
   avg.posVel[2] /= numAdded;
   avg.posVel[3] /= numAdded;
   //avg.visibility /= numAdded;
   avg.timeSinceVisible /= numAdded;
*/
   return avg;
}

ostream& operator<<(ostream& os, const ObjectState& s)
{
    os   << "xPos: " << s.getXPos()
         << "\t yPos: " << s.getYPos()
         << "\t xVel: " << s.getXVel()
         << "\t yVel: " << s.getYVel()
         << "\t vis: "  << s.getVisibility(0);
       //<< "\t numAdded: " << s.numAdded;

    return os;
}