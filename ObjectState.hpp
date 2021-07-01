#ifndef OBJECT_STATE_HPP
#define OBJECT_STATE_HPP

#include <vector>
#include <tuple>
#include <ostream>
#include <istream>
#include <iostream>

using namespace std;

// Note to self:
// Currently limiting visHistory to be at most 7 steps long
// Most memory-efficient way to generalize this I think would
// be to make visHistorySize a template parameter and make
// visHistory a statically-sized array of chars.
class ObjectState
{
  protected:
   unsigned char timeSinceVisible;
   char type;
   unsigned short id;
   /*
   float visibility = 0;
   size_t numAdded = 1;
   */
   float posVel[4];
   unsigned char visHistorySize;
   // One bit per bool value
   // Leading bit is always 1 to avoid meaningful formatting variables like EOF
   unsigned char visHistory;

  public:
   ObjectState() = default;
   ObjectState(float xPos, float yPos, size_t visHistorySize, int type, size_t id);
   ObjectState(float xPos, float yPos, float xVel, float yVel, const vector<bool>& visibilityHistory, int type, size_t id);
   ObjectState(istream& in, size_t propertiesPerObject);
   ObjectState(const ObjectState& other) = default;
   virtual ~ObjectState() = default;

   virtual float getXPos() const;
   virtual float getYPos() const;
   virtual float getXVel() const;
   virtual float getYVel() const;
   virtual bool getVisibility(size_t idx) const;
   virtual size_t getVisHistorySize() const;
   virtual int getType() const;
   virtual tuple<float, float> getPos() const;
   virtual float getProperty(size_t idx) const;
   virtual size_t getNumProperties() const;
   virtual size_t getID() const;
   virtual size_t getTimeSinceVisible() const;
   virtual void update(float xPos, float yPos, bool visible);
   virtual void save(ostream& out) const;

   virtual void confine();
   virtual tuple<float, float, float> addError(const tuple<float, float, float>&);

   // arithmatic operations
   virtual ObjectState square() const;
   virtual ObjectState average() const;
   friend ostream& operator<<(ostream& os, const ObjectState& s);
   ObjectState& operator+=(const ObjectState& rhs);
   ObjectState& operator-=(const ObjectState& rhs);

};

#endif